# tests/energy/utils/stac/test_base.py
"""Tests for defining STAC fields from operators' values (the definers' base class)."""

from collections.abc import Callable
from functools import partial

import pandas as pd
import pytest

from rbc.energy.utils.stac.base import FieldDefiner, normalize_keys
from rbc.energy.utils.stac.entity import EntityTypeDefiner
from rbc.energy.utils.stac.measurement import MeasurementTypeDefiner
from rbc.energy.utils.stac.status import StatusDefiner
from rbc.utils.errors import DataStructureError

STATUSES = StatusDefiner(
    {"Operativa": "active", "En Construcción": "under_construction"}
)


def test_map_compares_normalized_and_keeps_missing() -> None:
    """Happy path: spellings map alike (case, accents), missing values stay missing."""
    out = STATUSES.map(
        pd.Series(["OPERATIVA", "En Construccion", None], index=list("abc"))
    )
    assert out.name == "status"
    assert out.iloc[:2].to_dict() == {"a": "active", "b": "under_construction"}
    assert pd.isna(out["c"])


def test_map_raises_listing_all_unknown_values() -> None:
    """Failure path: every value without a STAC value is named in one error."""
    with pytest.raises(
        DataStructureError, match=r"energy:status: \['En Limbo', 'Oculto'\]"
    ):
        STATUSES.map(pd.Series(["Oculto", "Operativa", "En Limbo", None]))


def test_map_required_raises_for_missing_values_too() -> None:
    """Failure path: if required, a missing value is named as one without a map."""
    with pytest.raises(DataStructureError, match=r"\['<missing>', 'Oculto'\]"):
        STATUSES.map(pd.Series(["Oculto", "Operativa", None]), required=True)


@pytest.mark.parametrize(
    "make_definer",
    [StatusDefiner, MeasurementTypeDefiner, partial(EntityTypeDefiner, "ons")],
)
def test_map_to_a_value_the_field_lacks_raises(
    make_definer: Callable[[dict[str, str]], FieldDefiner],
) -> None:
    """Failure path: a map naming a value its field doesn't take fails on creation.

    Args:
        make_definer (Callable[[dict[str, str]], FieldDefiner]): Builds the field's
            definer from a map.
    """
    with pytest.raises(
        ValueError, match=r"Incorrect values for energy:.*\['sunshine'\]"
    ):
        make_definer({"Sol": "sunshine"})


def test_normalize_keys_raises_on_clashing_spellings() -> None:
    """Failure path: two spellings of one key mapping to different values raise."""
    with pytest.raises(ValueError, match="map to different values"):
        normalize_keys({"Operativa": "active", "OPERATIVA": "inactive"})
