"""ONS MAPPINGS."""

from rbc.energy.utils.stac.fuel import Fuel, RawFuel

# ---------------------------------------------------------------------------
# Coordinate Finding Definitions (additions/overrides for OPERATOR_METADATA)
# ---------------------------------------------------------------------------
# ======= FUEL TYPES =======
FUELTYPE_BASE_MAPPING = {
    "TÉRMICA": "thermal",
    "termeletrica": "thermal",
    "Óleo": "oil",
    "Óleo Diesel": "diesel oil",
    "Óleo Combustível": "fuel oil",
    "Carvão": "coal",
    "Gás": "gas",
    "Resíduos Industriais": "industrial waste",
    "NUCLEAR": "nuclear",
    "Nuclear": "nuclear",
    "Biomassa": "biomass",
    "Bioen": "bioenergy",
    "Bioenergia": "bioenergy",
    "HIDROELÉTRICA": "hydroelectric",
    "Hidrelétrica": "hydroelectric",
    "Hidráulica": "hydro",
    "EOLIELÉTRICA": "wind",
    "Eólica": "wind",
    "FOTOVOLTAICA": "solar",
    "Fotovoltaica": "solar",
}
FUELTYPE_MAPPING = FUELTYPE_BASE_MAPPING.copy()
for key, value in FUELTYPE_BASE_MAPPING.items():
    key = key.title()
    if key.endswith("a"):
        FUELTYPE_MAPPING[key[:-1] + "o"] = value
        FUELTYPE_MAPPING[key + "s"] = value


# ======= ENTITY NAMES =======
# GENERIC_UNIT_TOKENS = {
#     "bloco",
#     "conj.",
#     "conjunto"
# }

# Map: ONS-specific EGE terms → english translations
# Apply before fuzzy matching to properly resolve to the correct tokens.
# E.g. to prevent matching:
#      ONS' "Pequenas Centrais Hidrelétricas da Copel" &
#      OSM's "Casa de Força da Pequena Central Hidrelétrica Colino Dois"
EGE_NAME_BASE_TRANSLATIONS = {
    # overrides for multi-key fuel values
    "Diesel": "diesel oil",
    "Combustível": "fuel oil",
    "Industrial": "industrial",
    "Resíduos": "waste",
    # from Operator (ONS)
    "Central": "power plant",
    "Pequena": "small",
    "Gov": "operator",
    "Governador": "operator",
    # from Locator (OSM)
    "Casa": "house",
    "Força": "power",
    "Usina": "power station",
    "UHE": "hydroelectric power station",
    "Barragem": "dam",
}
EGE_NAME_TRANSLATIONS = EGE_NAME_BASE_TRANSLATIONS.copy()
for key, value in EGE_NAME_BASE_TRANSLATIONS.items():
    key = key.title()
    if len(key) > 3:
        if key.endswith("a"):
            EGE_NAME_TRANSLATIONS[key + "s"] = value
        elif key.endswith("r"):
            EGE_NAME_TRANSLATIONS[key + "es"] = value
        elif key.endswith("al"):
            EGE_NAME_TRANSLATIONS[key.removesuffix("l") + "is"] = value
        elif key.endswith("em"):
            EGE_NAME_TRANSLATIONS[key.removesuffix("m") + "ns"] = value
        elif key.endswith("el"):
            EGE_NAME_TRANSLATIONS[key.removesuffix("l") + "is"] = value


# ---------------------------------------------------------------------------
# STAC Definitions (the energy extension's vocabulary)
# ---------------------------------------------------------------------------
# ======= FUEL TYPES =======
# Map: ONS's (nom_tipousina, nom_tipocombustivel) → STAC fuel (s. FuelTypeDefiner)
FUELS_TO_STAC: dict[RawFuel, Fuel] = {
    ("HIDROELÉTRICA", "Hidráulica"): Fuel("hydro"),  # unknown if dam or river
    ("EOLIELÉTRICA", "Eólica"): Fuel("wind"),  # unknown if on- or offshore
    ("FOTOVOLTAICA", "Fotovoltaica"): Fuel("solar", "pv"),  # photovoltaic = PV
    ("NUCLEAR", "Nuclear"): Fuel("nuclear"),
    ("TÉRMICA", "Biomassa"): Fuel("biomass"),
    ("TÉRMICA", "Carvão"): Fuel("coal"),
    ("TÉRMICA", "Gás"): Fuel("gas"),
    ("TÉRMICA", "Óleo Combustível"): Fuel("oil", "fuel_oil"),
    ("TÉRMICA", "Óleo Diesel"): Fuel("oil", "diesel"),
    ("TÉRMICA", "Multi-Combustível Diesel/Óleo"): Fuel("oil"),  # diesel or fuel oil
    # --- industrial waste: pulp mills (=biomass) and steel mills (=process gases)
    ("TÉRMICA", "Resíduos Industriais"): Fuel("waste", "industrial"),
    # --- no specific type: some small groups of thermal plants (Peq Usinas Tipo III)
    ("TÉRMICA", "Outras Multi-Combustível"): Fuel("unknown"),
    # --- error in ONS: "Conj. Folha Larga" = "wind" EGE, but defined as (wind, pv) in 2020
    ("EOLIELÉTRICA", "Fotovoltaica"): Fuel("wind"),
}

# ======= ENTITY TYPES =======
# Map: ONS's 'cod_modalidadeoperacao' → default STAC entity type (before location & grouping)
ENTITY_TYPES_TO_STAC = {
    # single plants
    "TIPO I": "facility",
    "TIPO II-A": "facility",
    "TIPO II-B": "facility",
    "TIPO II-C": "facility",
    "TIPO III": "facility",
    # groups of plants reported together
    "Conjunto de Usinas": "portfolio",  # Tipo II-C plants in ONS's operating adjustments
    "Pequenas Usinas (Tipo III)": "portfolio",  # small plants without relation to ONS
    "Pequenas Usinas (MMGD)": "portfolio",  # micro/mini distributed generation, per area
}

# ======= MEASUREMENT TYPES =======
# Map: ONS's 'cod_modalidadeoperacao' → STAC measurement type (s. data_sources.md)
MEASUREMENT_TYPES_TO_STAC = {
    "TIPO I": "gross_output",
    "TIPO II-A": "gross_output",
    "TIPO II-B": "gross_output",
    "TIPO II-C": "gross_output",
    "Conjunto de Usinas": "gross_output",
    # plants without a relationship to ONS: forecasts, not measurements
    "TIPO III": "forecast_output",
    "Pequenas Usinas (Tipo III)": "forecast_output",
    # values no ONS source explains (distributed PV is behind the meter)
    "Pequenas Usinas (MMGD)": "unknown",
}
