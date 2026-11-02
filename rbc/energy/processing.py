"""Shared output layout of the energy data processors.

Every processor writes its operator's data in the same columns, so that the Zarr and STAC
steps can treat all operators alike:
- Zarr data: one row per energy generating entity (EGE) and timestamp = time series EG data,
- STAC data: one row per EGE, its columns named after the STAC energy extension.

Both identify an EGE by its global ID ('id', s. `global_ids`), the STAC item's ID.
A column an operator doesn't provide stays empty, i.e. its value is unknown.
"""

import json
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from shapely.geometry import mapping

from rbc.coordinates.mappings import SYSOP_REGION_COL
from rbc.coordinates.resources.natural_earth import RegionRegistry
from rbc.coordinates.utils.values import normalize_name

# Columns of the Zarr data: one row per EGE and timestamp
ZARR_COLUMNS = ["id", "timestamp", "value"]

# Columns of the STAC data: one row per EGE (all but 'id' from STAC energy extension)
STAC_COLUMNS = [
    "id",
    "entity_ids",
    "entity_name",
    "entity_type",
    "facility_name",
    "group_name",
    "operator",
    "fuel_type",
    "fuel_subtype",
    "hybrid_fuel_types",
    "generation_capacity",
    "storage_power",
    "status",
    "measurement_type",
    "lat",
    "lon",
    "region",
    "geometry",
]

# Map: coordinate finding's output columns → the EGE table's (the same for every operator)
LOCATION_COLUMNS = {SYSOP_REGION_COL: "region"}


# ------------------------------------------------------------------
# GENERAL HELPERS
# ------------------------------------------------------------------
def ege_ids(sysop: str, codes: pd.Series, names: pd.Series) -> pd.Series:
    """Give EGEs their global ID, keyed by their ID or else their name.

    The same identity coordinate finding dedupes by, so the Zarr and STAC data join.

    Args:
        sysop (str): Name of the system operator, e.g. "cen".
        codes (pd.Series): The EGEs' IDs (missing for some).
        names (pd.Series): The EGEs' names.

    Returns:
        pd.Series: The EGEs' global IDs, the same in the Zarr and the STAC data.
    """
    return global_ids(sysop, codes_as_text(codes).fillna(names.astype("string")))


def codes_as_text(codes: pd.Series) -> pd.Series:
    """Write codes as text, numbers however they were read (116, 116.0, '116' -> '116').

    Args:
        codes (pd.Series): The EGEs' IDs (missing for some), e.g. 116.0 or 'AMBA'.

    Returns:
        pd.Series: The IDs as text, missing where an EGE has none.
    """
    num = pd.to_numeric(codes, errors="coerce")
    whole = num.where(num % 1 == 0).astype("Int64").astype("string")
    return whole.fillna(codes.astype("string"))  # e.g. 'AMBA', '48W000000ABC1234'


def global_ids(sysop: str, keys: pd.Series) -> pd.Series:
    """Give each EGE a readable ID unique across all operators: '<sysop>-<its key>'.

    The key is written in lowercase ASCII with dashes (e.g. CEN's plant ID 1907 gives
    "cen-1907", a plant known only by its name "BESS Andes" gives "cen-bess-andes").
    It needs no stored state: the same key gets the same ID in every run, and an EGE
    added later can't take an existing one's.

    Args:
        sysop (str): Name of the system operator, e.g. "cen".
        keys (pd.Series): Each EGE's key, unique within its operator (e.g. its code).

    Returns:
        pd.Series: The EGEs' global IDs (e.g. "cen-1907"), missing without a key.

    Raises:
        ValueError: If two keys give the same ID (e.g. "BESS Andes", "BESS-ANDES").
    """
    lookup = {
        key: f"{sysop}-{normalize_name(str(key)).replace(' ', '-')}"
        for key in keys.dropna().unique()
    }
    seen: dict[str, object] = {}
    for key, gid in lookup.items():
        if gid in seen:
            raise ValueError(
                f"'{sysop}' EGE keys '{seen[gid]}' and '{key}' share the ID '{gid}'."
            )
        seen[gid] = key
    return keys.map(lookup)


def to_utc(
    timestamps: pd.Series,
    tz: str,
    dst: pd.Series,
    skipped: Literal["raise", "NaT"] = "raise",
) -> pd.Series:
    """Convert local timestamps to UTC, keeping every hour.

    The time zone applies its country's DST rules, past ones included. The hour a DST
    change repeats (clocks set back) is told apart by `dst`. An hour a DST change skips
    (clocks set forward) can't hold real data, so it raises instead of being guessed,
    unless the caller handles such rows itself (`skipped="NaT"`).

    Args:
        timestamps (pd.Series): Naive local timestamps, e.g. '2024-01-01 00:00:00'.
        tz (str): The operator's IANA time zone, e.g. "America/Santiago".
        dst (pd.Series): Whether each timestamp is daylight saving time. Only read where
            the local time is repeated, e.g. True for the first of its two rows.
        skipped (Literal["raise", "NaT"], optional): What a timestamp in an hour a DST
            change skips gets: an error, or NaT for the caller to handle. Defaults to
            "raise".

    Returns:
        pd.Series: The timestamps in UTC (tz-aware), none missing (unless skipped).

    Raises:
        ValueError: If a timestamp is missing or (by default) lies in an hour a DST
            change skips.
    """
    local = pd.to_datetime(timestamps)
    if missing := int(local.isna().sum()):
        raise ValueError(f"{missing} timestamps are missing.")

    aware = local.dt.tz_localize(
        tz, ambiguous=dst.to_numpy(dtype=bool), nonexistent="NaT"
    )
    if skipped == "raise" and (in_gap := aware.isna()).any():
        raise ValueError(
            f"{int(in_gap.sum())} timestamps lie in an hour a DST change skips ({tz}), "
            f"e.g. {sorted(local[in_gap].astype(str).unique())[:5]}."
        )
    return aware.dt.tz_convert("UTC")


def check_totals_kept(before: pd.Series, after: pd.Series, what: str) -> None:
    """Raise if a reshaping step changed any EGE's total, so no value was lost or added.

    Args:
        before (pd.Series): Each EGE's total before the step, indexed by its ID.
        after (pd.Series): Each EGE's total after the step, indexed by its ID.
        what (str): The data the totals are of, for the error, e.g. a file name.

    Raises:
        ValueError: If an EGE's totals differ (beyond float rounding).
    """
    before, after = before.align(after, fill_value=0.0)
    changed = ~np.isclose(before.to_numpy(float), after.to_numpy(float))
    if changed.any():
        diff = (after - before)[changed]
        raise ValueError(
            f"Totals changed for {int(changed.sum())} EGEs in {what}: "
            f"{diff.head(5).round(3).to_dict()}"
        )


def log_unjoined_ids(zarr_ids: set[str], stac_ids: pd.Series) -> None:
    """Log the EGE IDs only one of the Zarr and STAC data has, as both must have them all.

    Args:
        zarr_ids (set[str]): The distinct IDs in the Zarr data.
        stac_ids (pd.Series): The IDs in the STAC data.
    """
    stac = set(stac_ids.dropna())
    for unjoined, lacking in ((zarr_ids - stac, "STAC"), (stac - zarr_ids, "Zarr")):
        if unjoined:
            logger.error(
                f"{len(unjoined)} EGE IDs have no {lacking} data, e.g. "
                f"{sorted(unjoined)[:5]}"
            )


# ------------------------------------------------------------------
# Zarr HELPERS
# ------------------------------------------------------------------
def to_layout(
    df: pd.DataFrame, layout: list[str], renames: dict[str, str] | None = None
) -> pd.DataFrame:
    """Bring data into a layout's columns, dropping all others.

    Args:
        df (pd.DataFrame): The operator's data.
        layout (list[str]): The layout's columns, in order (e.g. STAC_COLUMNS).
        renames (dict[str, str], optional): Map of a column → layout column, for any not
            named after the layout yet. Defaults to None.

    Returns:
        pd.DataFrame: The data in the layout's columns, empty where not provided.
    """
    return df.rename(columns=renames or {}).reindex(columns=layout)


# ------------------------------------------------------------------
# STAC HELPERS
# ------------------------------------------------------------------
def entity_ids(df_ege: pd.DataFrame, own: dict[str, pd.Series]) -> pd.Series:
    """Collect each EGE's IDs in other systems as JSON (STAC's energy:entity_ids).

    Besides the operator's own IDs, an EGE gets the ID of the source that coordinate
    finding located it by. Not a sibling, as that ID is its donor's.

    Args:
        df_ege (pd.DataFrame): The EGEs as coordinate finding returned them.
        own (dict[str, pd.Series]): Map of the operator's ID system → each EGE's ID in it.

    Returns:
        pd.Series: Each EGE's IDs as a JSON object, missing where it has none.
    """
    # a locator's columns other than lat/lon/match_method exist only once it matched
    df = df_ege.reindex(
        columns=["match_method", "gem.id", "ppdb.id", "osm.id", "osm.type"]
    )
    method = df["match_method"].astype("string").fillna("")
    by = method.str.split("_").str[0].mask(method.str.endswith("_sibling"), "")
    osm_ids = (
        df["osm.type"].astype("string")
        + "/"
        + pd.to_numeric(df["osm.id"], errors="coerce").astype("Int64").astype("string")
    )
    # osm-powerplants' IDs are OSM's ('way/123'), PPM's a row number (todo: 'projectID')
    ppdb_ids = df["ppdb.id"].astype("string")
    ppdb_ids = ppdb_ids.where(
        ppdb_ids.str.fullmatch(r"(node|way|relation)/\d+", na=False)
    )

    ids = pd.DataFrame(
        {
            **own,
            "gem": df["gem.id"].astype("string").where(by == "gem"),
            "osm": ppdb_ids.where(by == "ppdb").fillna(osm_ids.where(by == "osm")),
        },
        index=df.index,
    )
    return pd.Series(
        [
            json.dumps(row.dropna().to_dict()) if row.notna().any() else None
            for _, row in ids.iterrows()
        ],
        index=df.index,
        dtype=object,
    )


def combine_fuels(fuels: pd.DataFrame) -> pd.DataFrame:
    """Combine each EGE's reported fuels into its STAC fuel fields.

    Args:
        fuels (pd.DataFrame): One row per EGE and reported fuel, with the columns "id",
            "fuel_type" and "fuel_subtype" (in STAC's vocabulary).

    Returns:
        pd.DataFrame: Indexed by EGE ID, with "fuel_type", "fuel_subtype" and
            "hybrid_fuel_types" (a list, only for hybrids).

    Raises:
        ValueError: If an EGE reports no fuel at all.
    """
    named = fuels.dropna(subset=["fuel_type"])
    if missing := sorted(set(fuels["id"]) - set(named["id"])):
        raise ValueError(f"{len(missing)} EGEs report no fuel type: {missing[:5]}")

    rows: dict[str, tuple[str, str | None, list[str] | None]] = {}
    for ege, group in named.groupby("id"):
        types = sorted(set(group["fuel_type"]))
        subtypes = sorted(set(group["fuel_subtype"].dropna()))
        if len(types) == 1:
            rows[ege] = (types[0], subtypes[0] if len(subtypes) == 1 else None, None)
        else:
            rows[ege] = ("hybrid", None, types)
    return pd.DataFrame.from_dict(
        rows,
        orient="index",
        columns=["fuel_type", "fuel_subtype", "hybrid_fuel_types"],
    )


def geometries(
    df: pd.DataFrame, country: str, region_reg: RegionRegistry | None
) -> pd.Series:
    """Give each EGE the smallest geometry its location is certain to, as GeoJSON.

    A point where coordinates were found, else its region's polygon where the region is
    known (s. Natural Earth), else its country's polygon.

    Args:
        df (pd.DataFrame): EGEs in the STAC_COLUMNS layout ("lat", "lon", "region").
        country (str): The operator's country, e.g. "Chile".
        region_reg (RegionRegistry | None): Natural Earth region lookup for polygons.

    Returns:
        pd.Series: Each EGE's geometry (GeoJSON dict), in the index of `df`.

    Raises:
        ValueError: If an EGE needs a polygon but none is found for its country.
    """
    shapes: dict = {}  # polygons are shared by many EGEs: look each up once

    def polygon(region: str | None) -> dict | None:
        if region not in shapes:
            geom = region_reg.geometry(country, region) if region_reg else None
            shapes[region] = mapping(geom) if geom is not None else None
        return shapes[region]

    out = []
    for lat, lon, region in zip(df["lat"], df["lon"], df["region"]):
        if pd.notna(lat) and pd.notna(lon):
            out.append({"type": "Point", "coordinates": [lon, lat]})
        elif pd.notna(region) and (geom := polygon(region)) is not None:
            out.append(geom)
        elif (geom := polygon(None)) is not None:
            out.append(geom)
        else:
            raise ValueError(f"No polygon found for '{country}' to place EGEs in.")
    return pd.Series(out, index=df.index, dtype=object)
