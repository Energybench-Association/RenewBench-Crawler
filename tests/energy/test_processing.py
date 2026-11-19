# tests/energy/test_processing.py
"""Tests for the energy processors' shared output layout."""

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest
from loguru import logger
from shapely.geometry import Polygon, mapping

from rbc.energy.processing import (
    STAC_COLUMNS,
    ZARR_COLUMNS,
    check_totals_kept,
    combine_fuels,
    ege_ids,
    entity_ids,
    geometries,
    global_ids,
    log_unjoined_ids,
    to_layout,
    to_utc,
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

    assert first.tolist() == [later[2], later[1]] == ["cen-116", "cen-bess-beta"]
    assert later.nunique() == 3


def test_global_ids_differ_between_operators() -> None:
    """Happy path: the same key of two operators gives two IDs, unique across all."""
    assert (
        global_ids("cen", pd.Series(["1"]))[0] != global_ids("ons", pd.Series(["1"]))[0]
    )


def test_global_ids_missing_without_key() -> None:
    """Failure path: an EGE without a key gets no ID, rather than a shared one."""
    assert global_ids("cen", pd.Series(["1", None])).isna().tolist() == [False, True]


def test_global_ids_raise_on_clash() -> None:
    """Failure path: two keys written the same way raise, so no two EGEs share an ID."""
    with pytest.raises(ValueError, match="share the ID 'cen-bess-beta'"):
        global_ids("cen", pd.Series(["BESS Beta", "BESS-BETA"]))


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
    "local, dst, utc",
    [
        ("2024-01-01 00:00:00", True, "2024-01-01 03:00:00"),  # summer: DST, UTC-3
        ("2025-08-01 00:00:00", True, "2025-08-01 04:00:00"),  # winter: UTC-4
        ("2023-04-01 23:00:00", True, "2023-04-02 02:00:00"),  # repeated: 1st, DST
        ("2023-04-01 23:00:00", False, "2023-04-02 03:00:00"),  # repeated: 2nd, std
        ("2015-06-01 00:00:00", True, "2015-06-01 03:00:00"),  # 2015: DST all year
    ],
)
def test_to_utc(local: str, dst: bool, utc: str) -> None:
    """Happy path: local time becomes UTC by the date's own offset, every hour kept.

    The hour clocks are set back is told apart by its DST flag, not left empty.

    Args:
        local (str): A local timestamp in Chile.
        dst (bool): Whether the row is the DST one of a repeated hour.
        utc (str): The expected UTC timestamp.
    """
    out = to_utc(pd.Series([local]), tz="America/Santiago", dst=pd.Series([dst]))
    assert out.tolist() == [pd.Timestamp(utc, tz="UTC")]


def test_to_utc_keeps_both_rows_of_the_repeated_hour() -> None:
    """Happy path: the day clocks are set back has 25 distinct UTC hours."""
    hours = [f"2023-04-01 {h:02d}:00:00" for h in range(24)] + ["2023-04-01 23:00:00"]
    dst = pd.Series(hours).duplicated().map({False: True, True: False})

    out = to_utc(pd.Series(hours), tz="America/Santiago", dst=dst)

    assert out.nunique() == 25
    assert out.min() == pd.Timestamp("2023-04-01 03:00", tz="UTC")  # 00:00, UTC-3
    assert out.max() == pd.Timestamp("2023-04-02 03:00", tz="UTC")  # 2nd 23:00, UTC-4


@pytest.mark.parametrize(
    "local", ["2023-09-03 00:00:00", None], ids=["skipped_hour", "missing"]
)
def test_to_utc_raises_instead_of_dropping(local: str | None) -> None:
    """Failure path: an hour clocks skip, or no timestamp, raises instead of vanishing.

    Args:
        local (str | None): A timestamp no UTC hour exists for.
    """
    with pytest.raises(ValueError):
        to_utc(pd.Series([local]), tz="America/Santiago", dst=pd.Series([True]))


def test_check_totals_kept_passes_equal_totals() -> None:
    """Happy path: totals equal up to float rounding pass silently."""
    check_totals_kept(
        pd.Series({"a": 0.1 + 0.2, "b": 5.0}), pd.Series({"a": 0.3, "b": 5.0}), "x"
    )


def test_check_totals_kept_raises_on_lost_values() -> None:
    """Failure path: an EGE whose values shrank or vanished raises with its ID."""
    with pytest.raises(ValueError, match="2 EGEs in day.csv"):
        check_totals_kept(
            pd.Series({"a": 3.0, "b": 5.0}), pd.Series({"a": 2.0}), "day.csv"
        )


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


def test_geometries_give_the_smallest_certain_geometry() -> None:
    """Happy path: a point where coordinates were found, else the region, else the country.

    A region with no polygon of that name falls back to the country, so no EGE is unplaced.
    """
    region, country = (
        Polygon([(0, 0), (1, 0), (1, 1)]),
        Polygon([(0, 0), (5, 0), (5, 5)]),
    )
    reg = MagicMock()
    reg.geometry.side_effect = lambda _c, r=None: {None: country, "Biobío": region}.get(
        r
    )
    df = pd.DataFrame(
        {
            "lat": [-37.3, None, None, None],
            "lon": [-72.5, None, None, None],
            "region": ["Biobío", "Biobío", "Atlántida", None],
        }
    )

    out = geometries(df, country="Chile", region_reg=reg)

    assert out[0] == {"type": "Point", "coordinates": [-72.5, -37.3]}
    assert out[1] == mapping(region)
    assert out[2] == out[3] == mapping(country)


def test_geometries_raise_without_a_country_polygon() -> None:
    """Failure path: an EGE that can't be placed at all raises, rather than lacking one."""
    reg = MagicMock()
    reg.geometry.return_value = None
    df = pd.DataFrame({"lat": [None], "lon": [None], "region": [None]})
    with pytest.raises(ValueError, match="No polygon"):
        geometries(df, country="Chile", region_reg=reg)


@pytest.mark.parametrize(
    "fuels, expected",
    [
        ([("coal", None)], ("coal", None, None)),
        ([("hydro", "reservoir")], ("hydro", "reservoir", None)),
        ([("gas", "lng"), ("gas", None)], ("gas", "lng", None)),
        ([("gas", "lng"), ("gas", "lpg")], ("gas", None, None)),
        ([("gas", "lng"), ("oil", "diesel")], ("hybrid", None, ["gas", "oil"])),
        ([("gas", None), (None, "chp")], ("gas", None, None)),
    ],
)
def test_combine_fuels(fuels: list[tuple], expected: tuple) -> None:
    """Happy path: one fuel is the fuel type, several make a hybrid listing them.

    Args:
        fuels (list[tuple]): An EGE's reported (fuel_type, fuel_subtype) rows.
        expected (tuple): Its expected fuel_type, fuel_subtype, hybrid_fuel_types.
    """
    df = pd.DataFrame(
        [("a", *f) for f in fuels], columns=["id", "fuel_type", "fuel_subtype"]
    )
    out = combine_fuels(df)
    assert tuple(out.loc["a"]) == expected


def test_combine_fuels_raises_without_any_fuel() -> None:
    """Failure path: an EGE reporting no fuel at all raises, rather than lacking one."""
    df = pd.DataFrame({"id": ["a"], "fuel_type": [None], "fuel_subtype": ["chp"]})
    with pytest.raises(ValueError, match="no fuel type"):
        combine_fuels(df)
