# tests/energy/test_processing.py
"""Tests for the energy processors' shared output layout."""

import json
from unittest.mock import patch

import pandas as pd
import pytest
from loguru import logger

from rbc.energy.processing import (
    STAC_COLUMNS,
    ZARR_COLUMNS,
    ege_ids,
    entity_ids,
    geometry_level,
    global_ids,
    log_unjoined_ids,
    to_layout,
    to_utc_iso,
)


def test_to_layout_keeps_only_the_layouts_columns() -> None:
    """Happy path: renamed columns are kept in the layout's order, all others dropped."""
    df = pd.DataFrame({"gen_real_mw": [1.0], "fecha_hora": ["2024-01-01"], "hora": [1]})

    out = to_layout(
        df,
        layout=ZARR_COLUMNS,
        renames={"fecha_hora": "timestamp", "gen_real_mw": "value"},
    )

    assert list(out.columns) == ZARR_COLUMNS
    assert out.loc[0, ["timestamp", "value"]].tolist() == ["2024-01-01", 1.0]


def test_to_layout_leaves_what_isnt_provided_unknown() -> None:
    """Happy path: a layout column the operator doesn't provide exists, but stays empty.

    So every operator's output has the same columns, whatever it reports.
    """
    out = to_layout(pd.DataFrame({"central": ["HE ANTUCO"]}), STAC_COLUMNS)

    assert list(out.columns) == STAC_COLUMNS
    assert out.loc[0].isna().all()


def test_global_ids_are_the_same_in_every_run() -> None:
    """Happy path: an EGE's global ID depends on its key alone, not on the other EGEs.

    So an EGE added later neither changes the existing IDs nor takes one of them.
    """
    first = global_ids("cen", pd.Series(["116", "BESS BETA"]))
    later = global_ids("cen", pd.Series(["2", "BESS BETA", "116"]))

    assert first.tolist() == [later[2], later[1]]
    assert later.nunique() == 3
    assert first.str.fullmatch(r"cen_[0-9a-f]{12}").all()


def test_global_ids_differ_between_operators() -> None:
    """Happy path: the same key of two operators gives two IDs, unique across all."""
    assert (
        global_ids("cen", pd.Series(["1"]))[0] != global_ids("ons", pd.Series(["1"]))[0]
    )


def test_global_ids_missing_without_key() -> None:
    """Failure path: an EGE without a key gets no ID, rather than a shared one."""
    assert global_ids("cen", pd.Series(["1", None])).isna().tolist() == [False, True]


def test_global_ids_raise_on_clash() -> None:
    """Failure path: two keys hashing to the same ID raise, so no two EGEs share one."""
    with patch("rbc.energy.processing.hashlib.sha256") as sha:
        sha.return_value.hexdigest.return_value = "0" * 64
        with pytest.raises(ValueError, match="share a global ID"):
            global_ids("cen", pd.Series(["1", "2"]))


@pytest.mark.parametrize(
    "code, name, key",
    [
        (116.0, "HE ANTUCO", "116"),
        ("116", "HE ANTUCO [NO_MOSTRAR]", "116"),
        ("AMBA", "ANGRA I", "AMBA"),  # e.g. ONS
        ("48W000000ABC1234", "UNIT 1", "48W000000ABC1234"),  # e.g. ENTSO-E's EIC
        (None, "BESS BETA", "BESS BETA"),
    ],
)
def test_ege_ids(code: float | str | None, name: str, key: str) -> None:
    """Happy path: an EGE's global ID is keyed by its code as text, else by its name.

    Both tables are written with these IDs, so they have to agree however a numeric code
    was read (116, 116.0, '116'), and a code that isn't a number is a code all the same.

    Args:
        code (float | str | None): The EGE's code, as read.
        name (str): Its name.
        key (str): The key its global ID should derive from.
    """
    expected = global_ids("cen", pd.Series([key])).tolist()
    assert ege_ids("cen", pd.Series([code]), pd.Series([name])).tolist() == expected


@pytest.mark.parametrize(
    "local, utc",
    [
        ("2024-01-01 00:00:00", "2024-01-01T03:00:00+00:00"),  # summer: DST, UTC-3
        ("2025-08-01 00:00:00", "2025-08-01T04:00:00+00:00"),  # winter: UTC-4
        ("2024-04-06 23:00:00", None),  # DST ends: this hour happens twice
        ("2024-09-08 00:00:00", None),  # DST starts: this hour is skipped
    ],
)
def test_to_utc_iso(local: str, utc: str | None) -> None:
    """Happy & failure path: local time becomes UTC by the date's own offset.

    An hour a DST change makes ambiguous or skips stays empty, rather than guessed.

    Args:
        local (str): A local timestamp in Chile.
        utc (str | None): The expected UTC timestamp, None if it can't be known.
    """
    out = to_utc_iso(pd.Series([local]), tz="America/Santiago")
    assert out.tolist() == [utc] if utc else out.isna().all()


@pytest.mark.parametrize(
    "zarr_ids, stac_ids, errors",
    [
        ({"a", "b"}, ["a", "b"], []),
        ({"a", "b"}, ["a"], ["1 EGE IDs have no STAC data, e.g. ['b']"]),
        ({"a"}, ["a", "c"], ["1 EGE IDs have no Zarr data, e.g. ['c']"]),
    ],
)
def test_log_unjoined_ids(
    zarr_ids: set[str], stac_ids: list[str], errors: list[str]
) -> None:
    """Happy & failure path: an ID only one of the Zarr and STAC data has is logged.

    Args:
        zarr_ids (set[str]): The distinct IDs in the Zarr data.
        stac_ids (list[str]): The IDs in the STAC data.
        errors (list[str]): The expected error messages.
    """
    captured_logs: list[str] = []
    sink_id = logger.add(lambda msg: captured_logs.append(msg.record["message"]))
    try:
        log_unjoined_ids(zarr_ids, pd.Series(stac_ids))
    finally:
        logger.remove(sink_id)

    assert captured_logs == errors


def _entity_ids(rows: list[dict]) -> list[dict | None]:
    """Collect the entity IDs of EGEs given as coordinate finding's output rows.

    Args:
        rows (list[dict]): The EGEs' columns, "cen" for the operator's own ID.

    Returns:
        list[dict | None]: Each EGE's IDs, parsed from JSON.
    """
    df = pd.DataFrame(rows)
    own = {"cen": df.pop("cen").astype("string")}
    return [json.loads(ids) if ids else None for ids in entity_ids(df, own)]


def test_entity_ids_name_the_source_that_located_the_ege() -> None:
    """Happy path: the operator's own ID, plus the locating source's, fuzzy or not.

    osm-powerplants' IDs (ppdb) are OSM's own, so they share the "osm" key.
    """
    out = _entity_ids(
        [
            {"cen": "1", "match_method": "official_id_exact"},
            {"cen": "2", "match_method": "gem_name_fuzzy", "gem.id": "G1"},
            {"cen": None, "match_method": "ppdb_id_parent_fuzzy", "ppdb.id": "way/5"},
            {
                "cen": "4",
                "match_method": "osm_name_exact",
                "osm.id": 7.0,
                "osm.type": "node",
            },
        ]
    )
    assert out == [
        {"cen": "1"},
        {"cen": "2", "gem": "G1"},
        {"osm": "way/5"},
        {"cen": "4", "osm": "node/7"},
    ]


def test_entity_ids_skip_ids_that_name_another_ege() -> None:
    """Failure path: no ID from a match that wasn't kept, nor from a sibling's donor.

    A vetoed match keeps its locator columns in coordinate finding's output, a sibling
    inherits its donor's: both IDs belong to another EGE. PPM's 'id' is a row number.
    """
    out = _entity_ids(
        [
            {"cen": None, "match_method": "unmatched", "ppdb.id": "way/9"},
            {"cen": "2", "match_method": "gem_sibling", "gem.id": "G1"},
            {"cen": None, "match_method": "ppdb_id_exact", "ppdb.id": "123"},
        ]
    )
    assert out == [None, {"cen": "2"}, None]


def test_entity_ids_without_any_locators_columns() -> None:
    """Happy path: a run no locator matched in lacks their columns, which isn't an error."""
    assert _entity_ids([{"cen": "1", "match_method": "unmatched"}]) == [{"cen": "1"}]


def test_geometry_level_names_the_smallest_certain_geometry() -> None:
    """Happy path: a point where coordinates were found, else the region, else the country."""
    df = pd.DataFrame(
        {
            "lat": [-37.3, None, None],
            "region": ["Biobío", "Coquimbo", None],
        }
    )
    assert geometry_level(df).tolist() == ["point", "region", "country"]
