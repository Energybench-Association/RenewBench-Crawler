"""STAC FIELD DEFINERS: BASE CLASS.

Shared base for defining a STAC energy field from an operator's own values (s. e.g.
`StatusDefiner`), and normalizing an operator's map keys (also for `FuelTypeDefiner`).
"""

from collections.abc import Hashable, Mapping
from typing import Any, ClassVar, TypeVar

import pandas as pd

from rbc.coordinates.utils.values import normalize_name
from rbc.utils.errors import DataStructureError

V = TypeVar("V")


class FieldDefiner:
    """Base class for defining a specific STAC field from an operator's value(s).

    Child classes provide the STAC field (`FIELD`) and the values (`VALUES`) it may assume.

    Attributes:
        FIELD (str): The field's column, e.g. "status" (STAC's energy:status).
        VALUES (frozenset[str]): The values the field may take.
        mapping (dict[Hashable, str]): Map of a normalized operator value → STAC's.
    """

    FIELD: ClassVar[str]
    VALUES: ClassVar[frozenset[str]]

    def __init__(self, mapping: Mapping[str, str]) -> None:
        """Initialize the definer with a mapping of operator's raw values to STAC's values.

        Args:
            mapping (Mapping[str, str]): Map of an operator's value → STAC's.

        Raises:
            ValueError: If the map names a value the field doesn't accept (s. VALUES), so
                a typo fails before any file is processed.
        """
        if wrong := sorted(set(mapping.values()) - self.VALUES):
            raise ValueError(f"Incorrect values for energy:{self.FIELD}: {wrong}")

        self.mapping = normalize_keys(mapping)

    def map(self, values: pd.Series, required: bool = False) -> pd.Series:
        """Map each operator value to STAC's, a missing one staying missing.

        Args:
            values (pd.Series): The operator's values.
            required (bool, optional): Whether a missing value is an error too.
                Defaults to False.

        Returns:
            pd.Series: STAC's values (named FIELD) with the values' index.

        Raises:
            DataStructureError: If values have no STAC value, listing all of them.
        """
        mapped = values.map(
            lambda v: self.mapping.get(normalize_key(v)) if isinstance(v, str) else None
        )

        unknown = values[mapped.isna() & (values.notna() | required)]
        if unknown_values := sorted(set(unknown.fillna("<missing>")), key=str):
            raise DataStructureError(
                f"Values without a STAC energy:{self.FIELD}: {unknown_values}"
            )
        return mapped.rename(self.FIELD)


# ------------------------------------------------------------------
# GENERAL HELPERS
# ------------------------------------------------------------------
def normalize_key(key: str | tuple[str | None, ...] | None) -> Hashable:
    """Normalize a mapping key, a value or a tuple of values (s. `normalize_name`).

    Args:
        key (str | tuple[str | None, ...] | None): E.g. "Térmica" or ("Térmica", None).

    Returns:
        Hashable: The key without case, accents or punctuation.
    """
    if isinstance(key, tuple):
        return tuple(normalize_name(k) for k in key)
    return normalize_name(key)


def normalize_keys(mapping: Mapping[Any, V]) -> dict[Hashable, V]:
    """Normalize a mapping's keys, so that an operator's spellings of one value match.

    Args:
        mapping (Mapping[Any, V]): Map of an operator's value (a string or a tuple of
            them) → STAC's.

    Returns:
        dict[Hashable, V]: The same map, its keys normalized (s. `normalize_key`).

    Raises:
        ValueError: If two keys normalize alike but map to different values.
    """
    out: dict[Hashable, V] = {}

    for key, value in mapping.items():
        norm = normalize_key(key)
        if norm in out and out[norm] != value:
            raise ValueError(f"Keys spelled like {key!r} map to different values.")

        out[norm] = value

    return out
