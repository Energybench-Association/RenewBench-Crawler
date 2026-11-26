"""STAC FUEL FIELDS.

Operator-neutral rules for the STAC energy extension's fuel fields (energy:fuel_type,
energy:fuel_subtype, energy:hybrid_fuel_types), applied to an operator's EGEs before they
become STAC Items.
"""

from dataclasses import dataclass

import pandas as pd

from rbc.energy.utils.stac.base import normalize_key, normalize_keys
from rbc.utils.errors import DataStructureError

# Map: energy:fuel_type values STAC Items may be defined as → their fuel_subtype(s)
FUELS: dict[str, set[str]] = {
    "biomass": {"chp"},
    "coal": {"anthracite", "bituminous", "lignite"},
    "gas": {"biogas", "cng", "lng", "lpg"},
    "geothermal": set(),
    "hydro": {"reservoir", "run_of_river"},
    "nuclear": set(),
    "oil": {"diesel", "fuel_oil", "petcoke"},
    "other": {"thermal"},
    "peat": set(),
    "solar": {"csp", "pv"},
    "storage": set(),
    "waste": {"industrial"},
    "wasteheat": {"chp"},
    "wind": {"offshore", "onshore"},
}


@dataclass(frozen=True)
class Fuel:
    """One fuel in STAC's vocabulary (s. FUELS), checked on creation."""

    type: str
    subtype: str | None = None

    def __post_init__(self) -> None:
        """Raise if the fuel isn't in FUELS: a typo fails on import, not on publishing.

        Raises:
            ValueError: If the type, or the subtype for this type, isn't in FUELS.
        """
        if self.type not in FUELS or (
            self.subtype is not None and self.subtype not in FUELS[self.type]
        ):
            raise ValueError(f"{self} isn't a fuel in FUELS.")


RawFuel = tuple[str, str | None]  # Sysops's original reported fuel (type, subtype)
Fuels = Fuel | tuple[Fuel, ...]  # One STAC fuel or several (a hybrid)


class FuelTypeDefiner:
    """Define an operator's EGEs' STAC fuel fields from their reported fuels (s. FUELS).

    Unlike a `FieldDefiner` subclass, this reads two sysop columns (fuel and subfuel) and all
    of an EGE's rows, and returns three fields. Same as `FieldDefiner`, the values (fuels)
    are checked on creation (s. `Fuel`). All values are normalized (no case, accents or
    punctuation) before being compared, as operators may spell the same fuel in several ways.

    Attributes:
        fuels (dict[Hashable, tuple[Fuel, ...]]): Map of normalized reported fuels
            (type, subtype) → its STAC fuels.
        fuels_per_ege (dict[Hashable, dict[str, tuple[Fuel, ...]]]): Map of normalized
            reported non-classifiable fuels (type, subtype) → each EGE's STAC fuels by its ID.
    """

    def __init__(
        self,
        fuels: dict[RawFuel, Fuels],
        fuels_per_ege: dict[RawFuel, dict[str, Fuels]] | None = None,
    ) -> None:
        """Initialize the definer with the operator's fuels.

        Args:
            fuels (dict[RawFuel, Fuels]): Map of reported, classifiable fuels (type, subtype)
                → their STAC equivalents. If there are several "type"s, the EGE is a hybrid.
            fuels_per_ege (dict[RawFuel, dict[str, Fuels]] | None, optional): Nested map of
                reported non-classifiable fuels (type, subtype) → the EGE's global IDs and
                their specific STAC fuels. This is for fuels that aren't proper types in
                themselves (e.g. cogeneration) and EGEs that report them. Defaults to None.
        """
        self.fuels = normalize_keys(
            {raw: self._as_tuple(f) for raw, f in fuels.items()}
        )
        self.fuels_per_ege = normalize_keys(
            {
                raw: {id: self._as_tuple(f) for id, f in f_by_ege.items()}
                for raw, f_by_ege in (fuels_per_ege or {}).items()
            }
        )

    def map(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get each EGE's STAC fuel fields from its reported fuels.

        Each EGE has at least one fuel type (with a fuel subtype if it reports one). If it
        has multiple fuel types (e.g. an EGE with a diesel and a gas configuration), it's
        defined as a hybrid.

        Args:
            df (pd.DataFrame): The EGE rows with "id", "fuel_type" and "fuel_subtype" columns,
                as the operator reports them (several rows per EGE).

        Returns:
            pd.DataFrame: The dataframe indexed by EGE ID with STAC values in columns
                "fuel_type", "fuel_subtype" and "hybrid_fuel_types" (=list; only for hybrids).

        Raises:
            DataStructureError: If a reported fuel has no mapping (or an EGE none of its own),
                listing all of them.
        """
        rows: list[tuple[str, str, str | None]] = []
        unknown: set[tuple] = set()
        cols = ["id", "fuel_type", "fuel_subtype"]
        for ege_id, main, sub in df[cols].drop_duplicates().itertuples(index=False):
            raw = (
                main if isinstance(main, str) else None,
                sub if isinstance(sub, str) else None,
            )
            key = normalize_key(raw)
            if key in self.fuels_per_ege:
                fuels = self.fuels_per_ege[key].get(ege_id)
                if fuels is None:
                    unknown.add((*raw, ege_id))  # as the operator spells it
                    continue
            elif key in self.fuels:
                fuels = self.fuels[key]
            else:
                unknown.add(raw)
                continue
            rows.extend((ege_id, fuel.type, fuel.subtype) for fuel in fuels)

        if unknown:
            listed = sorted(unknown, key=str)
            raise DataStructureError(f"Fuels without a STAC energy:fuel_type: {listed}")

        return self._combine(pd.DataFrame(rows, columns=cols))

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _combine(fuels: pd.DataFrame) -> pd.DataFrame:
        """Combine each EGE's STAC fuels into its fuel fields.

        Args:
            fuels (pd.DataFrame): One row per EGE and STAC fuel, with "id", "fuel_type"
                and "fuel_subtype" columns.

        Returns:
            pd.DataFrame: The dataframe indexed by EGE ID with "fuel_type", "fuel_subtype"
                and "hybrid_fuel_types" (s. `map`).
        """
        rows: dict[str, tuple[str, str | None, list[str] | None]] = {}
        for ege_id, group in fuels.groupby("id"):
            types = sorted(set(group["fuel_type"]))
            subtypes = sorted(set(group["fuel_subtype"].dropna()))
            if len(types) == 1:
                subtype = subtypes[0] if len(subtypes) == 1 else None
                rows[ege_id] = (types[0], subtype, None)
            else:
                rows[ege_id] = ("hybrid", None, types)

        return pd.DataFrame.from_dict(
            rows,
            orient="index",
            columns=["fuel_type", "fuel_subtype", "hybrid_fuel_types"],
        )

    @staticmethod
    def _as_tuple(fuels: Fuels) -> tuple[Fuel, ...]:
        """Turn fuel(s) into a tuple of one or several Fuel items (several = hybrid).

        Args:
            fuels (Fuels): The STAC fuels - a single one or several (already as a tuple).

        Returns:
            tuple[Fuel, ...]: The STAC fuels as a tuple.
        """
        return (fuels,) if isinstance(fuels, Fuel) else fuels
