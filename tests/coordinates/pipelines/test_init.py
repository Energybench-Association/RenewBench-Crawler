# tests/coordinates/pipelines/test_init.py
"""Tests for the pipelines package's __init__ helpers."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from rbc.coordinates import pipelines
from rbc.coordinates.load import Run
from rbc.coordinates.pipelines import build_shared_resources, make_pipeline
from rbc.coordinates.pipelines.default import DefaultPipeline
from rbc.coordinates.pipelines.entsoe import EntsoePipeline
from rbc.coordinates.resources.eic import EICCodeRegistry
from rbc.coordinates.resources.gem import GEMLocator
from rbc.coordinates.resources.natural_earth import RegionRegistry
from rbc.coordinates.resources.osmpp import OSMPPLocator
from rbc.coordinates.resources.overpass import OverpassLocator
from rbc.coordinates.resources.ppm import PPMLocator


# ----------------------------------
# Fixtures
# ----------------------------------
@pytest.fixture
def eia_run(make_run: Callable[..., Run]) -> Run:
    """An "eia" run (default pipeline).

    Args:
        make_run (Callable[..., Run]): Factory for runs (s. conftest).

    Returns:
        Run: An (empty) EIA run.
    """
    return make_run("eia")


@pytest.fixture
def entsoe_run(make_run: Callable[..., Run]) -> Run:
    """An "entsoe" run (entsoe pipeline), in a known bidding zone.

    Args:
        make_run (Callable[..., Run]): Factory for runs (s. conftest).

    Returns:
        Run: An (empty) ENTSO-E run of the "10YNL----------L" zone.
    """
    return make_run("entsoe", name="1h_10YNL----------L", zone="10YNL----------L")


# ----------------------------------
# Tests: make_pipeline
# ----------------------------------
class TestMakePipeline:
    """Test the 'make_pipeline' function."""

    @pytest.mark.parametrize("eic_reg", [None, object()])
    def test_resolves_default_pipeline(self, eia_run: Run, eic_reg: Any) -> None:
        """Happy path: any operator other than entsoe resolves to DefaultPipeline.

        Providing an EIC registry should have no effect, as this can't be used by the
        default pipeline.

        Args:
            eia_run (Run): An "eia" run.
            eic_reg (Any): The EIC registry object (if any).
        """
        pipeline = make_pipeline(run=eia_run, eic_reg=eic_reg)
        assert isinstance(pipeline, DefaultPipeline)

    def test_resolves_entsoe_pipeline(self, entsoe_run: Run):
        """Happy path: the entsoe operator resolves to EntsoePipeline.

        Args:
            entsoe_run (Run): An "entsoe" run of the "10YNL----------L" zone.
        """
        pipeline = make_pipeline(run=entsoe_run)
        assert isinstance(pipeline, EntsoePipeline)

    @pytest.mark.parametrize("run_fix", ["eia_run", "entsoe_run"])
    def test_use_osm_loc(self, run_fix: str, request: pytest.FixtureRequest) -> None:
        """Happy path: both pipelines use the given (shared) Overpass locator.

        If a pipeline built its own locator instead, every directory would load its
        country again, and `--update` would be ignored.

        Args:
            run_fix (str): Name of the run fixture (default or entsoe).
            request (pytest.FixtureRequest): Pytest-provided request to get the fixture.
        """
        osm_loc = OverpassLocator()
        pipeline = make_pipeline(run=request.getfixturevalue(run_fix), osm_loc=osm_loc)
        assert pipeline.osm_loc is osm_loc

    @pytest.mark.parametrize("run_fix", ["eia_run", "entsoe_run"])
    def test_use_region_reg(self, run_fix: str, request: pytest.FixtureRequest) -> None:
        """Happy path: both pipelines use the given (shared) region index.

        If a pipeline built its own index instead, every directory would read the
        admin-1 data again, and `resources_dir` / `--update` would be ignored.

        Args:
            run_fix (str): Name of the run fixture (default or entsoe).
            request (pytest.FixtureRequest): Pytest-provided request to get the fixture.
        """
        region_reg = RegionRegistry()
        pipeline = make_pipeline(
            run=request.getfixturevalue(run_fix), region_reg=region_reg
        )
        assert pipeline.region_reg is region_reg


class TestBuildSharedResources:
    """Test the 'build_shared_resources' function."""

    def test_default_pipeline(self) -> None:
        """Happy path: default pipeline builds a OSMPP ppdb, a GEM loc but no EIC registry."""
        shared = build_shared_resources(sysop="eia", resources_dir=None)
        assert isinstance(shared.ppdb_loc, OSMPPLocator)
        assert isinstance(shared.gem_loc, GEMLocator)
        assert isinstance(shared.osm_loc, OverpassLocator)
        assert shared.eic_reg is None

    def test_entsoe_pipeline(self) -> None:
        """Happy path: entsoe pipeline builds a PPM ppdb, a GEM loc and an EIC registry."""
        shared = build_shared_resources(sysop="entsoe", resources_dir=None)
        assert isinstance(shared.ppdb_loc, PPMLocator)
        assert isinstance(shared.gem_loc, GEMLocator)
        assert isinstance(shared.osm_loc, OverpassLocator)
        assert shared.eic_reg is not None
        assert isinstance(shared.eic_reg, EICCodeRegistry)

    @pytest.mark.parametrize("update", [False, True])
    def test_resources_get_own_subfolders(self, tmp_path: Path, update: bool) -> None:
        """Happy path: each resource gets its own subfolder of `resources_dir`.

        The update flag has to reach every one of them, since `-u` refreshes all
        downloaded resources.

        Args:
            tmp_path (Path): Pytest-provided temporary directory (`resources_dir`).
            update (bool): Whether fresh copies were requested.
        """
        shared = build_shared_resources(
            sysop="entsoe", resources_dir=tmp_path, update=update
        )

        cast(MagicMock, pipelines.GEMLocator).assert_called_once_with(
            gem_dir=Path(tmp_path, "gem"), update=update
        )
        cast(MagicMock, pipelines.PPMLocator).assert_called_once_with(
            cache_dir=Path(tmp_path, "ppm"), update=update
        )
        cast(MagicMock, pipelines.EICCodeRegistry).assert_called_once_with(
            cache_dir=Path(tmp_path, "eic"), update=update
        )
        assert shared.osm_loc is not None
        assert shared.osm_loc.cache_dir == Path(tmp_path, "overpass")
        assert shared.osm_loc.update is update
        assert shared.region_reg is not None
        assert shared.region_reg.cache_dir == Path(tmp_path, "natural_earth")
        assert shared.region_reg.update is update

    def test_default_pipeline_caches_osmpp(self, tmp_path: Path) -> None:
        """Happy path: the default pipeline's OSMPP locator gets the `osmpp/` subfolder.

        Args:
            tmp_path (Path): Pytest-provided temporary directory (`resources_dir`).
        """
        build_shared_resources(sysop="eia", resources_dir=tmp_path)

        cast(MagicMock, pipelines.OSMPPLocator).assert_called_once_with(
            cache_dir=Path(tmp_path, "osmpp"), update=False
        )
