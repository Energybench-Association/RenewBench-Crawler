"""STAC FUEL FIELDS.

Operator-neutral rules for the STAC energy extension's fuel types (energy:fuel_type,
energy:fuel_subtype, energy:hybrid_fuel_types), applied to an operator's EGEs before they
become STAC Items.
"""

import pandas as pd


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
    with_type = fuels.dropna(subset=["fuel_type"])
    if missing := sorted(set(fuels["id"]) - set(with_type["id"])):
        raise ValueError(f"{len(missing)} EGEs report no fuel type: {missing[:5]}")

    rows: dict[str, tuple[str, str | None, list[str] | None]] = {}
    for ege_id, group in with_type.groupby("id"):
        types = sorted(set(group["fuel_type"]))
        subtypes = sorted(set(group["fuel_subtype"].dropna()))
        if len(types) == 1:
            rows[ege_id] = (types[0], subtypes[0] if len(subtypes) == 1 else None, None)
        else:
            rows[ege_id] = ("hybrid", None, types)
    return pd.DataFrame.from_dict(
        rows,
        orient="index",
        columns=["fuel_type", "fuel_subtype", "hybrid_fuel_types"],
    )
