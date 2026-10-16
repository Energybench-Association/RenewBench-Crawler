"""Shared output layout of the energy data processors.

Every processor writes its operator's data in the same columns, so that the Zarr and STAC
steps can treat all operators alike:
- Zarr data: one row per energy generating entity (EGE) and timestamp = time series EG data,
- STAC data: one row per EGE, its columns named after the STAC energy extension.

Both identify an EGE by its global ID ('id', s. `global_ids`), the STAC item's ID.
A column an operator doesn't provide stays empty, i.e. its value is unknown.
"""

import hashlib
import json

import pandas as pd
from loguru import logger

from rbc.coordinates.mappings import SYSOP_REGION_COL

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
    "generation_capacity",
    "status",
    "measurement_type",
    "lat",
    "lon",
    "region",
    "geometry_level",
    "location_method",
]

# Map: coordinate finding's output columns → the EGE table's (the same for every operator)
LOCATION_COLUMNS = {SYSOP_REGION_COL: "region", "match_method": "location_method"}


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
    """Give each EGE an ID unique across all operators: '<sysop>_<hash of its key>'.

    Hashing needs no stored state: the same key gets the same ID in every run, and an EGE
    added later can't take an existing one's.

    Args:
        sysop (str): Name of the system operator, e.g. "cen".
        keys (pd.Series): Each EGE's key, unique within its operator (e.g. its code).

    Returns:
        pd.Series: The EGEs' global IDs (e.g. "cen_e8bdbb4a2d39"), missing without a key.

    Raises:
        ValueError: If two keys hash to the same ID.
    """
    lookup = {
        key: f"{sysop}_{hashlib.sha256(str(key).encode()).hexdigest()[:12]}"
        for key in keys.dropna().unique()
    }
    if len(set(lookup.values())) < len(lookup):
        raise ValueError(
            f"Two '{sysop}' EGE keys share a global ID: lengthen the hash."
        )
    return keys.map(lookup)


def to_utc_iso(timestamps: pd.Series, tz: str) -> pd.Series:
    """Convert local timestamps to UTC in ISO 8601 (e.g. '2024-01-01T03:00:00+00:00').

    The time zone applies its country's DST rules, past ones included. A local time a
    DST change makes ambiguous or skips stays empty: which hour it means is unknown.

    Args:
        timestamps (pd.Series): Naive local timestamps, e.g. '2024-01-01 00:00:00'.
        tz (str): The operator's IANA time zone, e.g. "America/Santiago".

    Returns:
        pd.Series: The timestamps in UTC as text, missing where ambiguous or skipped.
    """
    local = pd.to_datetime(timestamps).dt.tz_localize(
        tz, ambiguous="NaT", nonexistent="NaT"
    )
    if unknown := int((local.isna() & timestamps.notna()).sum()):
        logger.warning(f"{unknown} timestamps at a DST change left empty ({tz}).")
    return local.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


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


def geometry_level(df: pd.DataFrame) -> pd.Series:
    """Name the smallest geometry each EGE's location is certain to.

    Args:
        df (pd.DataFrame): EGEs in the STAC_COLUMNS layout.

    Returns:
        pd.Series: "point" where coordinates were found, else "region" where the region
            is known, else "country".
    """
    level = pd.Series("country", index=df.index, dtype=object)
    level = level.mask(df["region"].notna(), "region")
    return level.mask(df["lat"].notna(), "point")
