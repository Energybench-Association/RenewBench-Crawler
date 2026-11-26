# tests/energy/utils/stac/test_geometry.py
"""Tests for the STAC Items' geometry."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from shapely.geometry import Polygon, mapping

from rbc.energy.utils.stac.geometry import geometries


def test_geometries_give_the_smallest_certain_geometry() -> None:
    """Happy path: a point where coordinates were found, else the region, else the country.

    A region with no polygon of that name falls back to the country, so no EGE is unplaced.
    """
    region, country = (
        Polygon([(0, 0), (1, 0), (1, 1)]),
        Polygon([(0, 0), (5, 0), (5, 5)]),
    )
    region_reg = MagicMock()
    region_reg.geometry.side_effect = lambda _c, r=None: {
        None: country,
        "Biobío": region,
    }.get(r)
    df = pd.DataFrame(
        {
            "lat": [-37.3, None, None, None],
            "lon": [-72.5, None, None, None],
            "region": ["Biobío", "Biobío", "Atlántida", None],
        }
    )

    out = geometries(df, country="Chile", region_reg=region_reg)

    assert out[0] == {"type": "Point", "coordinates": [-72.5, -37.3]}
    assert out[1] == mapping(region)
    assert out[2] == out[3] == mapping(country)


def test_geometries_raise_without_a_country_polygon() -> None:
    """Failure path: an EGE that can't be placed at all raises, rather than lacking one."""
    region_reg = MagicMock()
    region_reg.geometry.return_value = None
    df = pd.DataFrame({"lat": [None], "lon": [None], "region": [None]})
    with pytest.raises(ValueError, match="No polygon"):
        geometries(df, country="Chile", region_reg=region_reg)
