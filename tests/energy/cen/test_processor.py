# tests/energy/cen/test_processor.py
"""Tests for the CEN processor: CEN's own registry coordinates and the processing run."""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from rbc.coordinates.pipelines import SharedResources
from rbc.coordinates.resources.natural_earth import RegionRegistry
from rbc.energy.cen import processor
from rbc.energy.cen.processor import CODE_COL, NAME_COL, REGISTRY_RENAMES, CenProcessor


# ----------------------------------
# Fixtures
# ----------------------------------
@pytest.fixture
def registry_rows() -> pd.DataFrame:
    """CEN's registry as `CenCoordinateRegistry` hands it over (API names, lat/lon added).

    Returns:
        pd.DataFrame: Four EGEs, the last one without coordinates.
    """
    return pd.DataFrame(
        {
            "id_central": [1, 2, 3, 4],
            "central": ["TER ALPHA", "PMGD PFV DOÑA RUBENA", "HE QUILLECO", "PE GAMMA"],
            "tipo_tecnologia": ["Termoeléctrica", "Solar", "Hidroeléctrica", "Eólica"],
            "estado": ["Operativa"] * 4,
            "comuna": ["Renca", "Pencahue", "Quilleco", "Taltal"],
            "region": ["Metropolitana de Santiago", "Maule", "Biobío", "Antofagasta"],
            "lat": [-33.4, -35.4, -37.4, None],
            "lon": [-70.7, -71.8, -71.9, None],
        }
    )


@pytest.fixture
def make_processor(
    tmp_path: Path, registry_rows: pd.DataFrame
) -> Callable[..., CenProcessor]:
    """Factory for processors whose registry and shared resources are stand-ins.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.
        registry_rows (pd.DataFrame): CEN's registry, as handed over.

    Returns:
        Callable[..., CenProcessor]: Builds a processor, optionally with a region registry.
    """

    def _make(region_reg: RegionRegistry | None = None) -> CenProcessor:
        shared = SharedResources(
            gem_loc=None,
            ppdb_loc=None,
            osm_loc=None,
            region_reg=region_reg,
            eic_reg=None,
        )
        with (
            patch.object(
                processor,
                "CenCoordinateRegistry",
                return_value=SimpleNamespace(df_registry=registry_rows),
            ),
            patch.object(processor, "build_shared_resources", return_value=shared),
        ):
            return CenProcessor(
                token="token",
                input_dir=Path(tmp_path, "raw", "cen"),
                resources_dir=Path(tmp_path, "resources"),
                output_dir=Path(tmp_path, "processed", "cen"),
            )

    return _make


def _region_reg(levels: dict[str, str]) -> RegionRegistry:
    """A region registry classifying each registry point by its region's name only.

    Args:
        levels (dict[str, str]): Map of region → classification ("within_borders" else).

    Returns:
        RegionRegistry: The stand-in.
    """
    return cast(
        RegionRegistry,
        SimpleNamespace(
            classify_match=lambda country, region, coord: levels.get(
                region, "within_borders"
            )
        ),
    )


def _ege_rows(*rows: tuple[float | None, str | None]) -> pd.DataFrame:
    """A run's distinct EGE rows, given as (code, name).

    Args:
        *rows (tuple[float | None, str | None]): The EGEs' codes and names.

    Returns:
        pd.DataFrame: The rows, in CEN's own column names.
    """
    return pd.DataFrame(
        {
            CODE_COL: [code for code, _ in rows],
            NAME_COL: [name for _, name in rows],
            "tipo_tecnologia": "Térmica",
            "subtipo_tecnologia": None,
        }
    )


def _write_raw(raw_dir: Path) -> None:
    """Write two days of raw CEN generation data, one row per EGE and hour.

    Args:
        raw_dir (Path): The operator's raw "1h" directory.
    """
    raw_dir.mkdir(parents=True)
    for day in ("2024-01-01", "2024-01-02"):
        rows = _ege_rows((1.0, "TER ALPHA"), (None, "BESS BETA"))
        pd.concat([rows.assign(hour=h, generation=10.0 * h) for h in range(3)]).to_csv(
            Path(raw_dir, f"{day}.csv"), index=False
        )


# ----------------------------------
# Tests: the registry
# ----------------------------------
def test_registry_columns_renamed(make_processor: Callable) -> None:
    """Happy path: the registry keeps the API's names in its cache, the processor its own.

    Args:
        make_processor (Callable): Factory for processors.
    """
    proc = make_processor()
    assert set(REGISTRY_RENAMES.values()) <= set(proc.df_registry.columns)


def test_join_registry_matches_all_id_formats(make_processor: Callable) -> None:
    """Happy path: an ID matches whether it's read as 28, 28.0 or '28'.

    The raw code column is left as is, so the output keeps the IDs CEN wrote, and runs
    stay comparable with those made without the registry.

    Args:
        make_processor (Callable): Factory for processors.
    """
    df = pd.DataFrame({CODE_COL: [1.0, "2", None]})

    out = make_processor()._join_registry(df, CODE_COL, ["name"])

    assert out["official.name"].tolist()[:2] == ["TER ALPHA", "PMGD PFV DOÑA RUBENA"]
    assert pd.isna(out.loc[2, "official.name"])
    assert out[CODE_COL].tolist()[:2] == [1.0, "2"]


# ----------------------------------
# Tests: official coordinates
# ----------------------------------
def test_get_coordinates_become_official(make_processor: Callable) -> None:
    """Happy path: an EGE found by ID gets the registry's point as official coordinates.

    EGEs without registry coordinates (or without an ID) are kept with empty ones, for
    the pipeline to match.

    Args:
        make_processor (Callable): Factory for processors.
    """
    df = _ege_rows((1.0, "TER ALPHA"), (4.0, "PE GAMMA"), (None, "BESS BETA"))

    out = make_processor()._get_coordinates(df).set_index(NAME_COL)

    assert out.at["TER ALPHA", "official.lat"] == -33.4
    assert out.at["TER ALPHA", "official.match_method"] == "official_id_exact"
    assert out.loc[["PE GAMMA", "BESS BETA"], "official.lat"].isna().all()
    assert out.loc[["PE GAMMA", "BESS BETA"], "official.match_method"].isna().all()
    assert out.at["TER ALPHA", "official.region"] == "Metropolitana de Santiago"
    assert "official.name" not in out.columns  # only needed for the name check


@pytest.mark.parametrize(
    "cen_name, code",
    [
        ("TER ALPHA [NO_MOSTRAR]", 1.0),  # a marker the registry doesn't carry
        ("PMGD PVF DOÑA RUBENA", 2.0),  # a typo: ratio 95
    ],
)
def test_get_coordinates_match_stays_if_alike_names(
    make_processor: Callable, cen_name: str, code: float
) -> None:
    """Happy path: names differing only by a marker or a typo still confirm the ID.

    Args:
        make_processor (Callable): Factory for processors.
        cen_name (str): CEN's name for the EGE.
        code (float): Its ID.
    """
    out = make_processor()._get_coordinates(_ege_rows((code, cen_name)))
    assert out.loc[0, "official.lat"] == pytest.approx(-33.4 if code == 1 else -35.4)


def test_get_coordinates_one_alike_name_is_enough(make_processor: Callable) -> None:
    """Happy path: an ID is doubted only if none of its name variants is alike enough.

    Args:
        make_processor (Callable): Factory for processors.
    """
    df = _ege_rows((3.0, "HE QUILLECO"), (3.0, "HE RUCUE"))

    out = make_processor()._get_coordinates(df)

    assert out["official.lat"].notna().all()


def test_get_coordinates_match_removed_if_unlike_names(
    make_processor: Callable,
) -> None:
    """Happy path: an ID whose names differ loses its registry point, on all its rows.

    The ID link is doubtful then: trusting its point could place the EGE where another
    plant is. Its unnamed rows lose it too, or the dedupe could pick one of them.

    Args:
        make_processor (Callable): Factory for processors.
    """
    df = _ege_rows((3.0, "HE RUCUE"), (3.0, None), (1.0, "TER ALPHA"))

    out = make_processor()._get_coordinates(df)

    assert out.loc[out[CODE_COL] == 3.0, "official.lat"].isna().all()
    assert out.loc[out[CODE_COL] == 3.0, "official.match_method"].isna().all()
    assert out.loc[out[CODE_COL] == 1.0, "official.lat"].notna().all()


@pytest.mark.parametrize("level, kept", [("mismatch", False), ("within_bounds", True)])
def test_get_coordinates_match_removed_if_outside_region(
    make_processor: Callable, level: str, kept: bool
) -> None:
    """Happy path: a registry point clearly outside the registry's own region is dropped.

    The record contradicts itself then (e.g. a mislabelled UTM zone put PFV TUTUVEN
    544 km away). A point just over the border (within the buffer) is kept.

    Args:
        make_processor (Callable): Factory for processors.
        level (str): How the stand-in classifies the point in "Biobío".
        kept (bool): Whether the point should be kept.
    """
    proc = make_processor(region_reg=_region_reg({"Biobío": level}))
    df = _ege_rows((3.0, "HE QUILLECO"), (1.0, "TER ALPHA"))

    out = proc._get_coordinates(df).set_index(NAME_COL)

    lat = out.at["HE QUILLECO", "official.lat"]
    assert lat == -37.4 if kept else pd.isna(lat)
    assert out.at["TER ALPHA", "official.lat"] == -33.4


def test_get_coordinates_match_kept_if_no_region(make_processor: Callable) -> None:
    """Happy path: without a region registry, registry points aren't region-checked.

    Args:
        make_processor (Callable): Factory for processors.
    """
    out = make_processor(region_reg=None)._get_coordinates(
        _ege_rows((3.0, "HE QUILLECO"))
    )
    assert out.loc[0, "official.lat"] == -37.4


# ----------------------------------
# Tests: the processing run
# ----------------------------------
def test_process_data(make_processor: Callable, tmp_path: Path) -> None:
    """Happy path: a run mirrors each raw CSV and describes each EGE once.

    The generation data is written per raw file, under its raw name: decades of daily
    CSVs would not fit into memory as one frame.

    Args:
        make_processor (Callable): Factory for processors.
        tmp_path (Path): Pytest-provided temporary directory.
    """
    _write_raw(Path(tmp_path, "raw", "cen", "1h"))
    proc = make_processor()

    with patch.object(processor, "build_map") as mock_map:
        proc.process_data()

    out_dir = Path(tmp_path, "processed", "cen")
    for day in ("2024-01-01", "2024-01-02"):
        mirrored = pd.read_csv(Path(out_dir, "1h", f"{day}.csv"))
        raw = pd.read_csv(Path(tmp_path, "raw", "cen", "1h", f"{day}.csv"))
        pd.testing.assert_frame_equal(mirrored, raw)

    coords = pd.read_csv(Path(out_dir, "coordinates", "coordinates_1h.csv"))
    assert sorted(coords["sysop.name"]) == ["BESS BETA", "TER ALPHA"]

    stac = pd.read_csv(Path(out_dir, "stac_data.csv")).set_index("sysop.name")
    assert stac.at["TER ALPHA", "official.status"] == "Operativa"
    assert stac.at["TER ALPHA", "match_method"] == "official_id_exact"

    assert mock_map.call_args.kwargs["labels"] == ["cen_1h"]


def test_process_data_without_raw(make_processor: Callable, tmp_path: Path) -> None:
    """Failure path: with no raw data, nothing is mapped or described, and nothing breaks.

    Args:
        make_processor (Callable): Factory for processors.
        tmp_path (Path): Pytest-provided temporary directory.
    """
    Path(tmp_path, "raw", "cen").mkdir(parents=True)
    proc = make_processor()

    with patch.object(processor, "build_map", MagicMock()) as mock_map:
        proc.process_data()

    mock_map.assert_not_called()
    assert not Path(tmp_path, "processed", "cen", "stac_data.csv").exists()
