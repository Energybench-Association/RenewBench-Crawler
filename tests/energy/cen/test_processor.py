# tests/energy/cen/test_processor.py
"""Tests for the CEN processor: CEN's own registry coordinates and the processing run."""

import json
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
from rbc.energy.cen.processor import (
    CODE_COL,
    NAME_COL,
    REGISTRY_RENAMES,
    CenProcessor,
)
from rbc.energy.processing import STAC_COLUMNS, ZARR_COLUMNS, global_ids


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


def _gid(key: str) -> str:
    """The global ID of a test EGE.

    Args:
        key (str): Its CEN ID as text, else its name.

    Returns:
        str: Its global ID.
    """
    return global_ids("cen", pd.Series([key]))[0]


def _raw_row(
    opreal: int,
    code: float | None,
    name: str,
    timestamp: str,
    generation: float,
    capacity: float,
    owner: str | None,
    fuel: str,
    subfuel: str,
) -> dict:
    """One raw row of CEN generation data (one configuration and hour).

    Args:
        opreal (int): The configuration's ID ('id_opreal').
        code (float | None): The plant's ID ('id_central'), missing for batteries.
        name (str): The plant's name ('central').
        timestamp (str): The hour's start ('fecha_hora').
        generation (float): The configuration's generation in MW ('gen_real_mw').
        capacity (float): The configuration's capacity in MW ('potencia_maxima').
        owner (str | None): The plant's owner ('propietario').
        fuel (str): Its technology ('tipo_tecnologia').
        subfuel (str): Its technology subtype ('subtipo_tecnologia').

    Returns:
        dict: The row, in CEN's own column names.
    """
    return {
        "id_opreal": opreal,
        CODE_COL: code,
        NAME_COL: name,
        "gen_real_mw": generation,
        "fecha_hora": timestamp,
        "potencia_maxima": capacity,
        "propietario": owner,
        "tipo_tecnologia": fuel,
        "subtipo_tecnologia": subfuel,
    }


def _write_raw(raw_dir: Path) -> None:
    """Write two days of raw CEN data, covering each way CEN reports an EGE (2 hours each).

    - TER ALPHA (ID 1): two configurations producing at once, one row repeated exactly
    - BESS BETA (no ID): a battery, discharging and charging, both reported positive
    - PE GAMMA (ID 4): one configuration, a status marker in its name, its capacity
      raised on the second day

    Args:
        raw_dir (Path): The operator's raw "1h" directory.
    """
    raw_dir.mkdir(parents=True)
    for day, gamma_capacity in (("2024-01-01", 30.0), ("2024-01-02", 32.0)):
        rows = []
        for hour in ("00:00:00", "01:00:00"):
            ts = f"{day} {hour}"
            rows += [
                _raw_row(
                    10, 1.0, "TER ALPHA", ts, 5.0, 50.0, "GEN SA", "Térmica", "Diésel"
                ),
                _raw_row(
                    11, 1.0, "TER ALPHA", ts, 7.0, 55.0, "GEN SA", "Térmica", "GNL"
                ),
                _raw_row(
                    20, None, "BESS BETA", ts, 4.0, 10.0, None, "BESS", "Inyección"
                ),
                _raw_row(21, None, "BESS BETA", ts, 3.0, 10.0, None, "BESS", "Retiro"),
                _raw_row(
                    40,
                    4.0,
                    "PE GAMMA [NO_MOSTRAR]",
                    ts,
                    2.0,
                    gamma_capacity,
                    "VIENTO SA",
                    "Eólica",
                    "Eólica",
                ),
            ]
        rows.append(rows[0])  # CEN repeats some rows exactly
        pd.DataFrame(rows).to_csv(Path(raw_dir, f"{day}.csv"), index=False)


@pytest.fixture
def processed(make_processor: Callable, tmp_path: Path) -> tuple[Path, MagicMock]:
    """Process the two days of raw CEN data.

    Args:
        make_processor (Callable): Factory for processors.
        tmp_path (Path): Pytest-provided temporary directory.

    Returns:
        tuple[Path, MagicMock]: The output directory, and the (mocked) map builder.
    """
    _write_raw(Path(tmp_path, "raw", "cen", "1h"))
    with patch.object(processor, "build_map") as mock_map:
        make_processor().process_data()
    return Path(tmp_path, "processed", "cen"), mock_map


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
def test_zarr_and_stac_ids_join(processed: tuple[Path, MagicMock]) -> None:
    """Happy path: the Zarr and STAC data describe the same EGEs, by the same IDs.

    The Zarr IDs come from the raw rows, the STAC IDs from coordinate finding's output.

    Args:
        processed (tuple[Path, MagicMock]): The output directory and the map builder.
    """
    out_dir, _ = processed
    zarr = pd.concat(pd.read_csv(p) for p in Path(out_dir, "1h").glob("*.csv"))
    stac = pd.read_csv(Path(out_dir, "stac_data.csv"))

    assert set(zarr["id"]) == set(stac["id"])
    assert zarr["id"].notna().all()


def test_generation_is_each_eges_net_output(processed: tuple[Path, MagicMock]) -> None:
    """Happy path: each raw CSV becomes one row per EGE and hour, in the shared layout.

    A plant's configurations producing at once are summed, a repeated row counts once,
    and a battery's charging (reported positive) is subtracted: the net output.

    Args:
        processed (tuple[Path, MagicMock]): The output directory and the map builder.
    """
    out_dir, _ = processed
    df = pd.read_csv(Path(out_dir, "1h", "2024-01-01.csv"))

    assert list(df.columns) == ZARR_COLUMNS
    first_hour = df[df["timestamp"] == "2024-01-01T03:00:00+00:00"].set_index("id")
    assert first_hour["value"].to_dict() == {
        _gid("1"): 12.0,
        _gid("4"): 2.0,
        _gid("BESS BETA"): 1.0,
    }


def test_ege_table_describes_each_ege_once(processed: tuple[Path, MagicMock]) -> None:
    """Happy path: the EGE table has the shared columns and one row per EGE.

    Its IDs are the generation data's, so both tables join.

    Args:
        processed (tuple[Path, MagicMock]): The output directory and the map builder.
    """
    out_dir, mock_map = processed
    df = pd.read_csv(Path(out_dir, "stac_data.csv"))

    assert list(df.columns) == STAC_COLUMNS
    assert sorted(df["id"]) == sorted([_gid("1"), _gid("4"), _gid("BESS BETA")])
    assert set(df["entity_type"]) == {"facility"}
    assert set(df["measurement_type"]) == {"gross_output"}
    assert mock_map.call_args.kwargs["labels"] == ["cen_1h"]


def test_ege_table_values(processed: tuple[Path, MagicMock]) -> None:
    """Happy path: each EGE's own values, as far as CEN provides them.

    - capacity: unknown for a plant with several generating configurations, a battery's
      one generating configuration's, and the latest a plant reported
    - status: a marker in the name overrides the registry's, and is dropped from the name
    - operator: CEN gives batteries none
    - entity IDs: CEN's own, where CEN gives one

    Args:
        processed (tuple[Path, MagicMock]): The output directory and the map builder.
    """
    out_dir, _ = processed
    df = pd.read_csv(Path(out_dir, "stac_data.csv")).set_index("id")
    alpha, gamma, beta = _gid("1"), _gid("4"), _gid("BESS BETA")

    assert pd.isna(df.at[alpha, "generation_capacity"])
    assert df.at[beta, "generation_capacity"] == 10.0
    assert df.at[gamma, "generation_capacity"] == 32.0

    assert df.at[alpha, "status"] == "Operativa"
    assert df.at[gamma, "status"] == "NO_MOSTRAR"
    assert df.at[gamma, "entity_name"] == "PE GAMMA"

    assert df.at[alpha, "operator"] == "GEN SA"
    assert pd.isna(df.at[beta, "operator"])
    assert df.at[alpha, "fuel_subtype"] == "Diésel, GNL"

    assert json.loads(df.at[gamma, "entity_ids"]) == {"cen": "4"}
    assert pd.isna(df.at[beta, "entity_ids"])


def test_ege_table_locations(processed: tuple[Path, MagicMock]) -> None:
    """Happy path: each EGE's location and how certain it is.

    Args:
        processed (tuple[Path, MagicMock]): The output directory and the map builder.
    """
    out_dir, _ = processed
    df = pd.read_csv(Path(out_dir, "stac_data.csv")).set_index("id")
    ids = [_gid("1"), _gid("4"), _gid("BESS BETA")]

    assert df.at[ids[0], "lat"] == -33.4
    assert df.at[ids[0], "location_method"] == "official_id_exact"
    assert df.loc[ids, "geometry_level"].tolist() == [
        "point",
        "region",  # in the registry without coordinates, but with its region
        "country",
    ]


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
