"""STAC GEOMETRY.

Operator-neutral rules for the STAC Items' geometry (GeoJSON, a core STAC field): an EGE's
point where its coordinates are known, else its region's or its country's polygon.
"""

import pandas as pd
from shapely.geometry import mapping

from rbc.coordinates.resources.natural_earth import RegionRegistry


def geometries(
    df_stac: pd.DataFrame, country: str, region_reg: RegionRegistry | None
) -> pd.Series:
    """Give each EGE the smallest geometry its location is certain to, as GeoJSON.

    A point where coordinates were found, else its region's polygon where the region is
    known (s. Natural Earth), else its country's polygon.

    Args:
        df_stac (pd.DataFrame): EGEs in the STAC_COLUMNS layout ("lat", "lon", "region").
        country (str): The operator's country, e.g. "Chile".
        region_reg (RegionRegistry | None): Natural Earth region lookup for polygons.

    Returns:
        pd.Series: Each EGE's geometry (GeoJSON dict), in the index of `df_stac`.

    Raises:
        ValueError: If an EGE needs a polygon but none is found for its country.
    """
    shapes: dict = {}  # polygons are shared by many EGEs: look each up once
    out = []
    for lat, lon, region in zip(df_stac["lat"], df_stac["lon"], df_stac["region"]):
        if pd.notna(lat) and pd.notna(lon):
            out.append({"type": "Point", "coordinates": [lon, lat]})
        elif (
            pd.notna(region)
            and (geom := _get_polygon(region, country, region_reg, shapes)) is not None
        ):
            out.append(geom)
        elif (geom := _get_polygon(None, country, region_reg, shapes)) is not None:
            out.append(geom)
        else:
            raise ValueError(f"No polygon found for '{country}' to place EGEs in.")
    return pd.Series(out, index=df_stac.index, dtype=object)


def _get_polygon(
    region: str | None, country: str, region_reg: RegionRegistry | None, shapes: dict
) -> dict | None:
    """Look up a region's polygon (or country's if None) as GeoJSON, once per region.

    Args:
        region (str | None): The region, e.g. "Biobío", or None for the whole country.
        country (str): The operator's country, e.g. "Chile".
        region_reg (RegionRegistry | None): Natural Earth region lookup for polygons.
        shapes (dict): The polygons looked up so far, by region; filled in place.

    Returns:
        dict | None: The polygon as GeoJSON, None if none is found.
    """
    if region not in shapes:
        geom = region_reg.geometry(country, region) if region_reg else None
        shapes[region] = mapping(geom) if geom is not None else None
    return shapes[region]
