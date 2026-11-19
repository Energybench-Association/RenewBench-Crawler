"""ONS MAPPINGS."""

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
# Map: ONS's (nom_tipousina, nom_tipocombustivel) → STAC (energy:fuel_type, energy:fuel_subtype)
STAC_FUELS: dict[tuple[str, str], tuple[str, str | None]] = {
    ("HIDROELÉTRICA", "Hidráulica"): ("hydro", None),  # ONS doesn't tell dam from river
    ("EOLIELÉTRICA", "Eólica"): ("wind", "onshore"),
    ("FOTOVOLTAICA", "Fotovoltaica"): ("solar", "pv"),
    ("NUCLEAR", "Nuclear"): ("nuclear", None),
    ("TÉRMICA", "Biomassa"): ("biomass", None),
    ("TÉRMICA", "Carvão"): ("coal", None),
    ("TÉRMICA", "Gás"): ("gas", None),
    ("TÉRMICA", "Óleo Combustível"): ("oil", "fuel_oil"),
    ("TÉRMICA", "Óleo Diesel"): ("oil", "diesel"),
    ("TÉRMICA", "Multi-Combustível Diesel/Óleo"): ("oil", None),  # diesel or fuel oil
    ("TÉRMICA", "Outras Multi-Combustível"): ("other", "thermal"),  # fuels not stated
    # pulp mills (black liquor) and steel mills (process gases); todo: per plant?
    ("TÉRMICA", "Resíduos Industriais"): ("waste", "industrial"),
}
# Map: ONS's combination of two technologies → their STAC fuels (a hybrid EGE)
STAC_HYBRID_FUELS: dict[tuple[str, str], list[tuple[str, str | None]]] = {
    # a wind plant type with a PV fuel (Conj. Folha Larga)
    ("EOLIELÉTRICA", "Fotovoltaica"): [("wind", "onshore"), ("solar", "pv")],
}

# ======= ENTITY TYPES =======
# ONS's 'cod_modalidadeoperacao' for single plants
PLANT_MODES = {"TIPO I", "TIPO II-A", "TIPO II-B", "TIPO II-C", "TIPO III"}
# ONS's 'cod_modalidadeoperacao' for groups of plants reported together
GROUP_MODES = {
    "Conjunto de Usinas",  # Tipo II-C plants grouped in ONS's operating adjustments
    "Pequenas Usinas (Tipo III)",  # small plants without a relationship to ONS
    "Pequenas Usinas (MMGD)",  # micro and mini distributed generation, per area
}
# ONS's modes of plants without a relationship to ONS: forecasts, not measurements (s.
# data_sources.md)
FORECAST_MODES = {"TIPO III", "Pequenas Usinas (Tipo III)"}
# ONS's modes whose values no ONS source explains (distributed PV is behind the meter)
UNKNOWN_MEASUREMENT_MODES = {"Pequenas Usinas (MMGD)"}
