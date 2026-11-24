# tests/energy/utils/stac/test_fuel.py
"""Tests for the energy extension's fuel fields."""

import pandas as pd
import pytest

from rbc.energy.utils.stac.fuel import combine_fuels


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
