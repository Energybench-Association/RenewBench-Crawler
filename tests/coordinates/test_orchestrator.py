# tests/coordinates/test_orchestrator.py
"""Tests for the coordinate/location finding orchestration functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from rbc.coordinates.load import Run
from rbc.coordinates.orchestrator import (
    _check_matchable,
    find_coordinates,
    perform_coordinate_finding,
)
from rbc.coordinates.pipelines import SharedResources

NO_RESOURCES = SharedResources(
    gem_loc=None, ppdb_loc=None, osm_loc=None, region_reg=None, eic_reg=None
)


# ----------------------------------
# Tests
# ----------------------------------
class TestFindCoordinates:
    """Tests for the find_coordinates function."""

    def test_find_coordinates(self) -> None:
        """Happy path: a Run df's rows come back as one row per EGE with the results."""
        df = pd.DataFrame(
            {
                "id_central": [1, 1, None],
                "central": ["TER ALPHA", "TER ALPHA", "BESS BETA"],
                "tipo_tecnologia": ["Térmica", "Térmica", "BESS"],
                "subtipo_tecnologia": ["Diésel", "Diésel", None],
            }
        )
        run = Run(sysop="cen", name="1h", zone=None, files=(), df=df)

        out = find_coordinates(run, shared=NO_RESOURCES)

        assert sorted(out["sysop.name"]) == ["BESS BETA", "TER ALPHA"]
        assert {"lat", "lon", "match_method"} <= set(out.columns)

    def test_find_coordinates_refuses_unmatchable(self) -> None:
        """Failure path: a run of a non-matchable operator is refused, like on the CLI."""
        run = Run(sysop="rei", name="1h", zone=None, files=(), df=pd.DataFrame())

        with pytest.raises(ValueError, match="can't be name-matched"):
            find_coordinates(run, shared=NO_RESOURCES)


class TestPerformCoordinateFinding:
    """Tests for the perform_coordinate_finding function."""

    def test_perform_coordinate_finding(self, tmp_path: Path) -> None:
        """Happy path: every directory becomes a run, and all of them go on one map.

        Args:
            tmp_path (Path): Pytest-provided temporary directory.
        """
        for tres in ("15min", "1h"):
            csv_dir = Path(tmp_path, "cen", tres)
            csv_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "id_central": [1],
                    "central": ["TER ALPHA"],
                    "tipo_tecnologia": ["Térmica"],
                    "subtipo_tecnologia": ["Diésel"],
                }
            ).to_csv(Path(csv_dir, "2024-01-01.csv"), index=False)

        with (
            patch(
                "rbc.coordinates.orchestrator.build_shared_resources",
                return_value=NO_RESOURCES,
            ),
            patch("rbc.coordinates.orchestrator.build_map") as mock_map,
        ):
            perform_coordinate_finding(sysop="cen", input_dirs=[tmp_path])

        assert len(mock_map.call_args.args[0]) == 2  # one EGE frame per run
        assert mock_map.call_args.kwargs["labels"] == ["cen_15min", "cen_1h"]

    def test_perform_coordinate_finding_refuses_unmatchable(self) -> None:
        """Failure path: the CLI entry refuses a non-matchable operator before loading."""
        build = MagicMock()
        with patch("rbc.coordinates.orchestrator.build_shared_resources", build):
            with pytest.raises(ValueError, match="can't be name-matched"):
                perform_coordinate_finding(sysop="eia", input_dirs=[])

        build.assert_not_called()


# ----------------------------------
# Tests - Helpers
# ----------------------------------
class TestHelperFunctions:
    """Tests for the helper functions (e.g. _check_matchable)."""

    @pytest.mark.parametrize("sysop", ["aemo", "eia", "rei"])
    def test_check_matchable_refuses_unmatchable(self, sysop: str) -> None:
        """Failure path: operators whose names don't identify single EGEs are refused.

        Their locations come from elsewhere (AEMO's data carries them, EIA reports companies,
        REI prefectures), so matching their names could only produce false positives.

        Args:
            sysop (str): A non-matchable operator.
        """
        with pytest.raises(ValueError, match="can't be name-matched"):
            _check_matchable(sysop)

    def test_check_matchable_refuses_unknown_operator(self) -> None:
        """Failure path: an operator without metadata is refused before anything runs."""
        with pytest.raises(ValueError, match="Unknown energy source"):
            _check_matchable("not_a_real_operator")
