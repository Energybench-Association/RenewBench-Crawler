# tests/energy/utils/stac/test_fuel.py
"""Tests for the energy extension's fuel fields."""

import pandas as pd
import pytest

from rbc.energy.utils.stac.fuel import Fuel, FuelTypeDefiner
from rbc.utils.errors import DataStructureError


@pytest.mark.parametrize(
    "fuels, expected",
    [
        ([("coal", None)], ("coal", None, None)),
        ([("hydro", "reservoir")], ("hydro", "reservoir", None)),
        ([("gas", "lng"), ("gas", None)], ("gas", "lng", None)),
        ([("gas", "lng"), ("gas", "lpg")], ("gas", None, None)),
        ([("gas", "lng"), ("oil", "diesel")], ("hybrid", None, ["gas", "oil"])),
    ],
)
def test_combine(fuels: list[tuple], expected: tuple) -> None:
    """Happy path: one fuel type is the EGE's, several make a hybrid listing them.

    Args:
        fuels (list[tuple]): An EGE's STAC (fuel_type, fuel_subtype) rows.
        expected (tuple): Its expected fuel_type, fuel_subtype, hybrid_fuel_types.
    """
    df = pd.DataFrame(
        [("a", *f) for f in fuels], columns=["id", "fuel_type", "fuel_subtype"]
    )
    out = FuelTypeDefiner._combine(df)
    assert tuple(out.loc["a"]) == expected


@pytest.mark.parametrize(
    "fuel_type, subtype",
    [("solar", "wind"), ("sunshine", None)],
    ids=["subtype_of_another_type", "unknown_type"],
)
def test_fuel_outside_the_vocabulary_raises(
    fuel_type: str, subtype: str | None
) -> None:
    """Failure path: a fuel not in FUELS raises on creation, so a typo fails on import.

    Args:
        fuel_type (str): The fuel type.
        subtype (str | None): Its subtype.
    """
    with pytest.raises(ValueError, match="isn't a fuel in FUELS"):
        Fuel(fuel_type, subtype)


DEFINER = FuelTypeDefiner(
    fuels={
        ("Térmica", "Carbón"): Fuel("coal"),
        ("Térmica", "Diésel"): Fuel("oil", "diesel"),
        ("Eólica", "Fotovoltaica"): (Fuel("wind", "onshore"), Fuel("solar", "pv")),
    },
    fuels_per_ege={("Térmica", "Cogeneración"): {"op-1": Fuel("biomass", "chp")}},
)


def _rows(*rows: tuple[str, str, str | None]) -> pd.DataFrame:
    """An operator's EGE rows with their reported fuels.

    Args:
        *rows (tuple[str, str, str | None]): Each row's ID, fuel type and subtype.

    Returns:
        pd.DataFrame: The rows with "id", "fuel_type" and "fuel_subtype".
    """
    return pd.DataFrame(rows, columns=["id", "fuel_type", "fuel_subtype"])


def test_fuel_definer_maps_spellings_hybrids_and_per_ege_fuels() -> None:
    """Happy path: spellings map alike, two fuel types make a hybrid, an EGE its own."""
    out = DEFINER.map(
        _rows(
            ("op-5", "TERMICA", "carbon"),
            ("op-5", "Térmica", "Carbón"),  # the same fuel again: once
            ("op-6", "Térmica", "Carbón"),  # configurations with two fuels: a hybrid
            ("op-6", "Térmica", "Diésel"),
            ("op-7", "Eólica", "Fotovoltaica"),  # one pair naming two fuels: a hybrid
            ("op-1", "Térmica", "Cogeneración"),
        )
    )
    assert out.fillna("-").to_dict(orient="index") == {
        "op-1": {
            "fuel_type": "biomass",
            "fuel_subtype": "chp",
            "hybrid_fuel_types": "-",
        },
        "op-5": {"fuel_type": "coal", "fuel_subtype": "-", "hybrid_fuel_types": "-"},
        "op-6": {
            "fuel_type": "hybrid",
            "fuel_subtype": "-",
            "hybrid_fuel_types": ["coal", "oil"],
        },
        "op-7": {
            "fuel_type": "hybrid",
            "fuel_subtype": "-",
            "hybrid_fuel_types": ["solar", "wind"],
        },
    }


def test_fuel_definer_raises_listing_all_unknown_fuels() -> None:
    """Failure path: unmapped fuels and EGEs lacking their own raise, all listed."""
    rows = _rows(
        ("op-5", "Mareomotriz", None),
        ("op-6", "Nuclear", "PWR"),
        ("op-9", "Térmica", "Cogeneración"),
    )
    with pytest.raises(DataStructureError) as error:
        DEFINER.map(rows)
    assert all(v in str(error.value) for v in ("Mareomotriz", "PWR", "op-9"))
