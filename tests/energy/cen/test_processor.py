# tests/energy/cen/test_processor.py
"""Tests for the CEN processor: CEN's own registry coordinates and the processing run."""

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import icechunk as ic
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Polygon

from rbc.config.schema import PublishConfig
from rbc.coordinates.pipelines import SharedResources
from rbc.coordinates.resources.natural_earth import RegionRegistry
from rbc.energy.cen import processor
from rbc.energy.cen.processor import (
    CODE_COL,
    NAME_COL,
    REGISTRY_RENAMES,
    CenProcessor,
)
from rbc.energy.processing import combine_fuels, global_ids

PUBLISH_CFG = PublishConfig(
    release="v0.1.0",
    data_base_url="https://data.example.org",
    storage={"endpoint": "https://s3.example.org"},
    stac_api="https://stac.example.org",
)
# Stand-in polygons of a region and of the country
REGION = Polygon([(-71, -38), (-70, -38), (-70, -37)])
COUNTRY = Polygon([(-75, -55), (-66, -55), (-66, -17)])


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

    def _make(
        region_reg: RegionRegistry | None = None,
        publish_cfg: PublishConfig = PUBLISH_CFG,
        upload: bool = False,
    ) -> CenProcessor:
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
                publish_cfg=publish_cfg,
                upload=upload,
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
            ),
            geometry=lambda country, region=None: REGION if region else COUNTRY,
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
    subfuel: str | None,
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
        subfuel (str | None): Its technology subtype ('subtipo_tecnologia'), if any.

    Returns:
        dict: The row, in CEN's own column names.
    """
    return {
        "id_opreal": opreal,
        CODE_COL: code,
        NAME_COL: name,
        "gen_real_mw": generation,
        "fecha_hora": timestamp,
        "hora": int(timestamp[11:13]) + 1,
        "potencia_maxima": capacity,
        "propietario": owner,
        "tipo_tecnologia": fuel,
        "subtipo_tecnologia": subfuel,
    }


def _write_raw(raw_dir: Path) -> None:
    """Write two days of raw CEN data, covering each way CEN reports an EGE (2 hours each).

    - TER ALPHA (ID 1): two configurations producing at once, one row repeated exactly,
      another repeated with the plant details of a later download
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
                    11, 1.0, "TER ALPHA", ts, 7.0, 55.0, "GEN SA", "Térmica", "GLP"
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
                    None,
                ),
            ]
        rows.append(rows[0])  # CEN repeats some rows exactly
        rows.append({**rows[1], "propietario": "GEN SPA"})  # or with later details
        pd.DataFrame(rows).to_csv(Path(raw_dir, f"{day}.csv"), index=False)


def _skip(*_args: object) -> None:
    """Stand in for full STAC validation (tested in tests/publish, as it fetches schemas).

    Args:
        *_args (object): The records that would be validated.
    """


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
    with (
        patch.object(processor, "build_map") as mock_map,
        patch.object(processor, "validate", _skip),
    ):
        make_processor(region_reg=_region_reg({})).process_data()
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
def _store(out_dir: Path) -> xr.Dataset:
    """Open the release's local repository at its tag.

    Args:
        out_dir (Path): The processed data directory.

    Returns:
        xr.Dataset: The release's generation data.
    """
    path = Path(out_dir, "icechunk", "energy", "cen", "PT1H", "repository")
    repo = ic.Repository.open(ic.local_filesystem_storage(str(path)))
    session = repo.readonly_session(tag=PUBLISH_CFG.release)
    return xr.open_zarr(session.store, consolidated=False).load()


def _items(out_dir: Path) -> dict[str, dict]:
    """Read the release's STAC Items, by ID.

    Args:
        out_dir (Path): The processed data directory.

    Returns:
        dict[str, dict]: The Items.
    """
    folder = Path(out_dir, "stac", "cen__v0.1.0", "items")
    return {p.stem: json.loads(p.read_text()) for p in folder.glob("*.json")}


def test_store_and_stac_ids_join(processed: tuple[Path, MagicMock]) -> None:
    """Happy path: the repository and the STAC Items describe the same EGEs, by ID.

    The repository's IDs come from coordinate finding, its values from the raw rows.

    Args:
        processed (tuple[Path, MagicMock]): The output directory and the map builder.
    """
    out_dir, _ = processed
    ds = _store(out_dir)

    assert set(ds["entity_id"].values) == set(_items(out_dir))
    assert ds["values"].notnull().any("time").all()


def test_generation_is_each_eges_net_output(processed: tuple[Path, MagicMock]) -> None:
    """Happy path: one value per EGE and UTC hour, every local hour kept.

    A plant's configurations producing at once are summed, a repeated row counts once,
    and a battery's charging (reported positive) is subtracted: the net output.

    Args:
        processed (tuple[Path, MagicMock]): The output directory and the map builder.
    """
    out_dir, _ = processed
    ds = _store(out_dir)

    first_hour = ds["values"].sel(time="2024-01-01T03:00")  # 00:00 local, UTC-3
    assert dict(zip(ds["entity_id"].values, first_hour.values.tolist())) == {
        _gid("1"): 12.0,
        _gid("4"): 2.0,
        _gid("BESS BETA"): 1.0,
    }
    assert ds["values"].attrs["units"] == "MW"
    assert int(ds["values"].notnull().sum()) == 3 * 4  # 3 EGEs, 2 hours on 2 days


def test_items_describe_each_ege_once(processed: tuple[Path, MagicMock]) -> None:
    """Happy path: one Item per EGE, in the energy extension's terms.

    Args:
        processed (tuple[Path, MagicMock]): The output directory and the map builder.
    """
    out_dir, mock_map = processed
    items = _items(out_dir)

    assert sorted(items) == sorted([_gid("1"), _gid("4"), _gid("BESS BETA")])
    props = [it["properties"] for it in items.values()]
    assert {p["energy:entity_type"] for p in props} == {"facility"}
    assert {p["energy:measurement_type"] for p in props} == {"gross_output"}
    assert mock_map.call_args.kwargs["labels"] == ["cen_1h"]


def test_item_values(processed: tuple[Path, MagicMock]) -> None:
    """Happy path: each EGE's own values, as far as CEN provides them.

    - capacity: unknown for a plant with several generating configurations, a battery's
      one generating configuration's as storage power, and the latest a plant reported
    - status: a marker in the name overrides the registry's, and is dropped from the
      name; both mapped to STAC's vocabulary
    - fuel: mapped to STAC's vocabulary, several fuels making a hybrid
    - operator: CEN gives batteries none
    - entity IDs: CEN's own, where CEN gives one

    Args:
        processed (tuple[Path, MagicMock]): The output directory and the map builder.
    """
    items = _items(processed[0])
    alpha, gamma, beta = (items[_gid(k)]["properties"] for k in ("1", "4", "BESS BETA"))

    assert "energy:generation_capacity" not in alpha
    assert beta["energy:storage_power"] == 10.0
    assert "energy:generation_capacity" not in beta
    assert gamma["energy:generation_capacity"] == 32.0

    assert alpha["energy:status"] == "active"  # the registry's "Operativa"
    assert gamma["energy:status"] == "decommissioned"  # "[NO_MOSTRAR]" overrides it
    assert gamma["energy:entity_name"] == "PE GAMMA"

    assert alpha["energy:fuel_type"] == "hybrid"
    assert alpha["energy:hybrid_fuel_types"] == ["gas", "oil"]
    assert (beta["energy:fuel_type"], gamma["energy:fuel_type"]) == ("storage", "wind")

    assert alpha["energy:operator"] == "GEN SA"
    assert "energy:operator" not in beta
    assert gamma["energy:entity_ids"] == {"cen": "4"}
    assert "energy:entity_ids" not in beta


def test_item_locations(processed: tuple[Path, MagicMock]) -> None:
    """Happy path: each EGE placed as precisely as its location is known.

    Args:
        processed (tuple[Path, MagicMock]): The output directory and the map builder.
    """
    items = _items(processed[0])
    alpha, gamma, beta = (items[_gid(k)] for k in ("1", "4", "BESS BETA"))

    assert alpha["geometry"] == {"type": "Point", "coordinates": [-70.7, -33.4]}
    assert alpha["bbox"] == [-70.7, -33.4, -70.7, -33.4]
    # in the registry without coordinates, but with its region; else the country
    assert gamma["bbox"] == list(REGION.bounds)
    assert beta["bbox"] == list(COUNTRY.bounds)


def test_items_span_their_data_and_point_at_it(
    processed: tuple[Path, MagicMock],
) -> None:
    """Happy path: an Item's time span is its data's, its asset the release's snapshot.

    Args:
        processed (tuple[Path, MagicMock]): The output directory and the map builder.
    """
    out_dir, _ = processed
    item = _items(out_dir)[_gid("4")]
    data = item["assets"]["data"]
    repo = ic.Repository.open(ic.local_filesystem_storage(data["href"]))

    assert item["properties"]["start_datetime"] == "2024-01-01T03:00:00Z"
    assert item["properties"]["end_datetime"] == "2024-01-02T05:00:00Z"
    assert data["icechunk:repository"] == "energy/cen/PT1H/repository"
    assert data["icechunk:entity_id"] == item["id"]
    assert data["icechunk:snapshot"] == repo.lookup_tag("v0.1.0")


def test_collection_describes_the_release(processed: tuple[Path, MagicMock]) -> None:
    """Happy path: the release's Collection summarizes its Items and states the terms.

    Args:
        processed (tuple[Path, MagicMock]): The output directory and the map builder.
    """
    path = Path(processed[0], "stac", "cen__v0.1.0", "collection.json")
    coll = json.loads(path.read_text())

    assert coll["id"] == "cen__v0.1.0"
    assert coll["summaries"]["energy:fuel_type"] == ["hybrid", "storage", "wind"]
    assert coll["links"][0]["href"] == processor.CEN_TERMS_URL
    assert "academic, non-commercial" in coll["description"]
    assert coll["extent"]["temporal"]["interval"] == [
        ["2024-01-01T03:00:00Z", "2024-01-02T05:00:00Z"]
    ]


def test_rerun_reuses_the_releases_tag(
    processed: tuple[Path, MagicMock], make_processor: Callable
) -> None:
    """Happy path: processing the same release again reuses its tag, writing nothing new.

    Args:
        processed (tuple[Path, MagicMock]): The output directory and the map builder.
        make_processor (Callable): Factory for processors.
    """
    out_dir, _ = processed
    before = _items(out_dir)[_gid("1")]["assets"]["data"]["icechunk:snapshot"]

    with (
        patch.object(processor, "build_map"),
        patch.object(processor, "validate", _skip),
    ):
        make_processor(region_reg=_region_reg({})).process_data()

    assert _items(out_dir)[_gid("1")]["assets"]["data"]["icechunk:snapshot"] == before


def test_publishing_refuses_placeholder_urls(make_processor: Callable) -> None:
    """Failure path: publishing with the public config's placeholders stops at once.

    The same config works without publishing, which needs none of the URLs.

    Args:
        make_processor (Callable): Factory for processors.
    """
    placeholders = PUBLISH_CFG.model_copy(
        update={"data_base_url": "YOUR-DATA-BASE-URL", "stac_api": "YOUR-STAC-API-URL"}
    )
    with pytest.raises(ValueError, match="data_base_url.*stac_api"):
        make_processor(publish_cfg=placeholders, upload=True)
    make_processor(publish_cfg=placeholders, upload=False)


def test_publishing_from_a_cluster(make_processor: Callable, tmp_path: Path) -> None:
    """Happy path: with `local_root` and `stac_inbox`, publishing needs no network.

    The repository is written straight to the shared filesystem, and the STAC records,
    still naming the public data URL, are handed over through the inbox instead of the
    STAC API. No S3 endpoint, credentials or API URL are involved.

    Args:
        make_processor (Callable): Factory for processors.
        tmp_path (Path): Pytest-provided temporary directory.
    """
    lsdf = Path(tmp_path, "lsdf")
    cfg = PUBLISH_CFG.model_copy(
        update={
            "storage": PUBLISH_CFG.storage.model_copy(
                update={
                    "endpoint": "YOUR-S3-ENDPOINT",
                    "local_root": Path(lsdf, "data"),
                }
            ),
            "stac_api": "YOUR-STAC-API-URL",
            "stac_inbox": Path(lsdf, "inbox"),
        }
    )
    _write_raw(Path(tmp_path, "raw", "cen", "1h"))
    with (
        patch.object(processor, "build_map"),
        patch.object(processor, "validate", _skip),
        patch.object(processor, "publish") as mock_publish,
    ):
        make_processor(
            region_reg=_region_reg({}), publish_cfg=cfg, upload=True
        ).process_data()

    prefix = "energy/cen/PT1H/repository"
    repo = ic.Repository.open(
        ic.local_filesystem_storage(str(Path(lsdf, "data", "energybench", prefix)))
    )
    folder = Path(lsdf, "inbox", "cen__v0.1.0")
    item = json.loads(next(Path(folder, "items").glob("*.json")).read_text())
    data = item["assets"]["data"]
    assert data["href"] == f"https://data.example.org/energybench/{prefix}"
    assert data["icechunk:snapshot"] == repo.lookup_tag("v0.1.0")
    assert Path(folder, "collection.json").is_file()
    mock_publish.assert_not_called()
    assert not Path(tmp_path, "processed", "cen", "icechunk").exists()


def test_process_data_without_raw(make_processor: Callable, tmp_path: Path) -> None:
    """Failure path: with no raw data, processing stops instead of publishing nothing.

    Args:
        make_processor (Callable): Factory for processors.
        tmp_path (Path): Pytest-provided temporary directory.
    """
    Path(tmp_path, "raw", "cen").mkdir(parents=True)
    with pytest.raises(ValueError, match="No raw 'cen' data"):
        make_processor().process_data()


# ----------------------------------
# Tests: DST and vocabulary
# ----------------------------------
def _day(day: str, hours: list[tuple[str, int]]) -> pd.DataFrame:
    """One configuration's rows of a local day, as `_prepare_for_zarr` gets them.

    Args:
        day (str): The local date.
        hours (list[tuple[str, int]]): The rows' local time ('HH:MM') and 'hora'.

    Returns:
        pd.DataFrame: The rows, renamed to the shared layout, 1 MW each.
    """
    return pd.DataFrame(
        {
            "id": "cen_x",
            "id_opreal": 1,
            "timestamp": [f"{day} {t}:00" for t, _ in hours],
            "hour": [h for _, h in hours],
            "value": 1.0,
            "fuel_subtype": "Carbón",
        }
    )


def _clock(day: str, horas: list[int]) -> list[tuple[str, int]]:
    """CEN's labels for its hours of a day: 'hora' k is labelled (k-1) % 24 o'clock.

    Args:
        day (str): The local date.
        horas (list[int]): The rows' 'hora'.

    Returns:
        list[tuple[str, int]]: Each row's 'fecha_hora' time ('HH:MM') and 'hora'.
    """
    return [(f"{(h - 1) % 24:02d}:00", h) for h in horas]


@pytest.mark.parametrize(
    "day, hours", [("2023-04-01", 25), ("2023-09-03", 23), ("2023-06-01", 24)]
)
def test_local_day_in_utc(day: str, hours: int) -> None:
    """Happy path: a local day in Chile lasts 25, 23 or 24 hours around DST changes.

    Args:
        day (str): A local date in Chile.
        hours (int): Its expected length in hours.
    """
    start, end = processor.local_day_in_utc(day)
    assert (end - start).total_seconds() / 3600 == hours


def test_fall_back_saturday_holds_the_first_repeated_hour(
    make_processor: Callable,
) -> None:
    """Happy path: Saturday's 23:00 before clocks are set back is DST (UTC-3).

    On 2023-04-01, Chile's clocks went from 24:00 (UTC-3) back to 23:00 (UTC-4).

    Args:
        make_processor (Callable): Factory for processors.
    """
    df = _day("2023-04-01", _clock("2023-04-01", list(range(1, 25))))
    out, _ = make_processor()._prepare_for_zarr(df, "2023-04-01")

    assert out["timestamp"].nunique() == 24
    assert out["timestamp"].max() == pd.Timestamp("2023-04-02 02:00", tz="UTC")


def test_fall_back_sunday_adds_the_repeated_hour_as_hora_25(
    make_processor: Callable,
) -> None:
    """Happy path: Sunday's 'hora' 25 (labelled 00:00) is the repeated 23:00, UTC-4.

    So Saturday's and Sunday's files together give every UTC hour exactly once.

    Args:
        make_processor (Callable): Factory for processors.
    """
    df = _day("2023-04-02", _clock("2023-04-02", list(range(1, 26))))
    out, placeholders = make_processor()._prepare_for_zarr(df, "2023-04-02")

    assert out["timestamp"].nunique() == 25
    assert out["timestamp"].min() == pd.Timestamp("2023-04-02 03:00", tz="UTC")
    assert out["timestamp"].max() == pd.Timestamp("2023-04-03 03:00", tz="UTC")
    assert out["value"].sum() == 25.0 and placeholders.empty


def test_spring_forward_placeholder_is_kept_out_and_returned(
    make_processor: Callable,
) -> None:
    """Happy path: the placeholder for the skipped hour is returned for review.

    Sunday's 'hora' 1 is labelled 00:00, a time the clocks skipped, so it isn't an hour.
    The 23 real hours become UTC.

    Args:
        make_processor (Callable): Factory for processors.
    """
    df = _day("2023-09-03", _clock("2023-09-03", list(range(1, 25))))
    df.loc[df["hour"] == 1, "value"] = 0.0
    out, placeholders = make_processor()._prepare_for_zarr(df, "2023-09-03")

    assert out["timestamp"].nunique() == 23
    assert out["timestamp"].min() == pd.Timestamp("2023-09-03 04:00", tz="UTC")
    assert placeholders["hour"].tolist() == [1]


@pytest.mark.parametrize(
    "day, rows",
    [
        ("2023-06-01", [("23:00", 24), ("00:00", 25)]),  # 'hora' 25 without a change
        ("2023-06-01", [("05:00", 6), ("05:00", 6)]),  # the same hour twice
        ("2023-06-01", [("05:00", 7)]),  # label disagrees with 'hora'
        ("2023-06-01", [("01:00", 26)]),  # no such 'hora'
    ],
    ids=["hora_25_no_change", "double_hour", "label_mismatch", "hora_26"],
)
def test_hours_that_cant_be_placed_raise(
    make_processor: Callable, day: str, rows: list[tuple[str, int]]
) -> None:
    """Failure path: an hour that can't be placed in UTC exactly once stops processing.

    Args:
        make_processor (Callable): Factory for processors.
        day (str): The local date.
        rows (list[tuple[str, int]]): The rows' local time and 'hora'.
    """
    df = _day(day, rows).assign(value=[1.0, 2.0][: len(rows)])
    with pytest.raises(ValueError):
        make_processor()._prepare_for_zarr(df, day)


# ----------------------------------
# Tests: repeated hours
# ----------------------------------
def test_repeated_hours_keep_the_first_rows() -> None:
    """Happy path: a configuration's hour held again with the same value is dropped.

    As in CEN's 31 December files, the repeat carries the plant details of a later
    download, so dropping identical rows leaves it in. The first copy's rows stay.
    """
    day = "2009-12-31"
    first = _day(day, _clock(day, [1, 2, 3])).assign(operator="GEN SA")
    again = _day(day, _clock(day, [2, 3])).assign(operator="GEN SPA")
    df = pd.concat([first, again], ignore_index=True)

    out = CenProcessor._drop_repeated_hours(df, day)

    assert out["hour"].tolist() == [1, 2, 3]
    assert (out["operator"] == "GEN SA").all()


def test_fall_back_hour_25_is_not_a_repeat() -> None:
    """Happy path: 'hora' 25 shares its 00:00 label with 'hora' 1, but is its own hour."""
    day = "2023-04-02"
    df = _day(day, _clock(day, list(range(1, 26))))

    assert len(CenProcessor._drop_repeated_hours(df, day)) == 25


def test_repeated_hour_in_doubt_leaves_the_eges_hour_out() -> None:
    """Happy path: copies that disagree, even by rounding, leave the EGE's hour out.

    Neither copy is known to be right. The EGE's other configurations go too, so its
    total for the hour is missing rather than too low; other EGEs keep the hour.
    """
    day = "2017-12-31"
    first = _day(day, _clock(day, [1, 2]))
    again = _day(day, _clock(day, [2])).assign(value=1.004)
    other_config = _day(day, _clock(day, [1, 2])).assign(id_opreal=2)
    other_ege = _day(day, _clock(day, [1, 2])).assign(id="cen_y", id_opreal=3)
    df = pd.concat([first, again, other_config, other_ege], ignore_index=True)

    out = CenProcessor._drop_repeated_hours(df, day)

    assert sorted(zip(out["id"], out["id_opreal"], out["hour"])) == [
        ("cen_x", 1, 1),
        ("cen_x", 2, 1),
        ("cen_y", 3, 1),
        ("cen_y", 3, 2),
    ]


@pytest.mark.parametrize(
    "main, sub, fuel",
    [
        ("Térmica", "Carbón", ("coal", None)),
        ("Hidráulica", "Embalse", ("hydro", "reservoir")),
        ("Solar", None, ("solar", "pv")),
        ("Térmica", "Termosolar", ("solar", "csp")),
        ("Térmica", "Cogeneración", ("biomass", "chp")),  # ID 1907: its own fuel
    ],
)
def test_stac_fuels(main: str, sub: str | None, fuel: tuple) -> None:
    """Happy path: CEN's (type, subtype) maps to STAC's fuel as listed in STAC_FUELS.

    Args:
        main (str): CEN's 'tipo_tecnologia'.
        sub (str | None): CEN's 'subtipo_tecnologia'.
        fuel (tuple): The expected (fuel_type, fuel_subtype).
    """
    df = pd.DataFrame(
        {"id": ["a"], CODE_COL: [1907.0], "fuel_type": [main], "fuel_subtype": [sub]}
    )
    out = CenProcessor._stac_fuels(df)
    assert tuple(out.loc[0, ["fuel_type", "fuel_subtype"]]) == fuel


def test_stac_fuels_ignore_case_and_accents() -> None:
    """Happy path: CEN's spellings of one technology map alike, one row per EGE.

    E.g. 'Cogeneracion' and 'Cogeneración' (CEN uses both), or 'TÉRMICA' and 'Térmica'.
    """
    df = pd.DataFrame(
        {
            "id": ["a", "a", "b"],
            CODE_COL: [1907.0, 1907.0, 5.0],
            "fuel_type": ["Térmica", "Termica", "TÉRMICA"],
            "fuel_subtype": ["Cogeneración", "Cogeneracion", "Carbón"],
        }
    )
    out = CenProcessor._stac_fuels(df)
    assert out.values.tolist() == [["a", "biomass", "chp"], ["b", "coal", None]]


def test_unmapped_vocabulary_raises() -> None:
    """Failure path: a type or status without a mapping stops processing, all listed."""
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            CODE_COL: [1.0, 2.0],
            "fuel_type": ["Mareomotriz", "Solar"],
            "fuel_subtype": None,
        }
    )
    with pytest.raises(ValueError, match="Mareomotriz"):
        CenProcessor._stac_fuels(df)
    # a cogeneration plant without its own fuel defined names none: combining raises
    cogen = pd.DataFrame(
        {
            "id": ["c"],
            CODE_COL: [999.0],
            "fuel_type": "Térmica",
            "fuel_subtype": "Cogeneración",
        }
    )
    with pytest.raises(ValueError, match="no fuel type"):
        combine_fuels(CenProcessor._stac_fuels(cogen))
    no_terms = pd.Series([None, None, None])
    with pytest.raises(ValueError, match="En Limbo"):
        CenProcessor._stac_status(no_terms, pd.Series(["Operativa", "En Limbo", None]))
    with pytest.raises(ValueError, match="OCULTO"):
        CenProcessor._stac_status(pd.Series(["OCULTO"]), pd.Series(["Operativa"]))


@pytest.mark.parametrize(
    "term, estado, status",
    [
        ("NO_MOSTRAR", "Operativa", "decommissioned"),
        ("No_Mostrar", "Fuera de Servicio", "decommissioned"),
        ("EN_REVISION", "En Pruebas", "under_construction"),
        (None, "En Pruebas", "active"),
        (None, None, None),
    ],
)
def test_stac_status(term: str | None, estado: str | None, status: str | None) -> None:
    """Happy path: the name's square bracket term decides, else the registry's estado.

    Args:
        term (str | None): The square bracket term in CEN's name.
        estado (str | None): The registry's status.
        status (str | None): The expected STAC status.
    """
    out = CenProcessor._stac_status(pd.Series([term]), pd.Series([estado]))
    assert out.tolist() == [status] if status else out.isna().all()
