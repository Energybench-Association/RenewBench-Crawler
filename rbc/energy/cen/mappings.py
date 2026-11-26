"""CEN MAPPINGS."""

from rbc.energy.utils.stac.fuel import Fuel, Fuels, RawFuel

# ---------------------------------------------------------------------------
# Coordinate Finding Definitions (additions/overrides for OPERATOR_METADATA)
# ---------------------------------------------------------------------------
# ======= FUEL TYPES =======
FUELTYPE_BASE_MAPPING = {
    # main types
    "Térmica": "thermal",
    "Nuclear": "nuclear",
    "Hidráulica": "hydro",
    "Geotérmica": "geothermal",
    "Eólica": "wind",
    "Solar": "solar",
    "BESS": "battery energy storage system",
    # subtypes for thermal
    "Diésel": "diesel oil",
    "Fuel Oil": "fuel oil",
    "Carbón": "coal",
    "Gas Natural": "natural gas",
    "BioGas": "biogas gas",
    "Biomasa": "biomass",
    "Cogeneración": "cogeneration, combined heat and power, CHP (gas, oil, diesel, biogas, biomass)",
    "Termosolar": "concentrated solar power, CSP",
    "PetCoke": "petroleum coke, petcoke",
    "GLP": "liquefied petroleum gas, LPG",
    # subtypes for hydro
    "Embalse": "hydro dam reservoir",
    "Pasada": "hydro river",
    # subtypes for BESS
    "Inyección": "battery storage discharge injection",
    "Retiro": "battery storage charge withdrawal",
}
FUELTYPE_MAPPING = FUELTYPE_BASE_MAPPING.copy()
for key, value in FUELTYPE_BASE_MAPPING.items():
    key = key.title()
    if key.endswith("a"):
        FUELTYPE_MAPPING[key[:-1] + "o"] = value
        FUELTYPE_MAPPING[key + "s"] = value


# ======= ENTITY NAMES =======
# Map: CEN-specific EGE terms → english translations
# Apply before fuzzy matching to properly resolve to the correct tokens.
EGE_NAME_BASE_TRANSLATIONS = {
    # from Operator (CEN)
    "los": "",
    "las": "",
    "del": "",  # article, like "los"/"las"
    "parque": "park",
    "TER": "thermal",  # térmica
    "GEO": "geothermal",  # geotérmica
    "HE": "hydro",  # hidroeléctrica
    "HP": "hydro river",  # hidroeléctrica de pasada
    "MCH": "mini hydro",  # mini central hidroeléctrica
    "PE": "wind farm",  # parque eólico
    "CSP": "concentrated solar power",  # concentración solar de potencia
    "PFV": "photovoltaic solar park",  # parque fotovoltaico
    "PVF": "photovoltaic solar park",  # typo for PFV (1 EGE)
    "PSF": "photovoltaic solar park",  # parque solar fotovoltaico
    "PVP": "photovoltaic solar park",
    "PMG": "small generator",  # pequeño medio de generación
    "PMGD": "small distributed generator",  # pequeño medio de generación distribuido
    "SAE": "storage",  # sistema de almacenamiento de energía
    # from Locators (GEM/OSMPP/OSM)
    "central": "power plant",
    "planta": "plant",
    "fotovoltaico": "photovoltaic solar",
    "fotovoltaica": "photovoltaic solar",
    "hidroelectrica": "hydro",
    "termoelectrica": "thermal",
}  # + FUELTYPE_BASE_MAPPING --> added together in coordinate finding (BasePipeline.__init__)
EGE_NAME_TRANSLATIONS = EGE_NAME_BASE_TRANSLATIONS.copy()
for key, value in EGE_NAME_BASE_TRANSLATIONS.items():
    key = key.title()
    if len(key) > 3:
        if key.endswith("a") or key.endswith("e"):
            EGE_NAME_TRANSLATIONS[key + "s"] = value
        elif key.endswith("r"):
            EGE_NAME_TRANSLATIONS[key + "es"] = value
        elif key.endswith("al"):
            EGE_NAME_TRANSLATIONS[key.removesuffix("l") + "es"] = value
        elif key.endswith("el"):
            EGE_NAME_TRANSLATIONS[key.removesuffix("l") + "is"] = value


# ---------------------------------------------------------------------------
# STAC Definitions (the energy extension's vocabulary)
# ---------------------------------------------------------------------------
# ======= FUEL TYPES =======
# Map: CEN's (tipo_tecnologia, subtipo_tecnologia) → STAC fuel (s. FuelTypeDefiner)
FUELS_TO_STAC: dict[RawFuel, Fuels] = {
    ("Solar", None): Fuel("solar", "pv"),
    ("Eólica", None): Fuel("wind", "onshore"),
    ("Geotérmica", None): Fuel("geothermal"),
    ("Hidráulica", "Embalse"): Fuel("hydro", "reservoir"),
    ("Hidráulica", "Pasada"): Fuel("hydro", "run_of_river"),
    ("BESS", "Inyección"): Fuel("storage"),  # discharging
    ("BESS", "Retiro"): Fuel("storage"),  # charging
    ("Térmica", "Carbón"): Fuel("coal"),
    ("Térmica", "Gas Natural"): Fuel("gas"),
    ("Térmica", "GLP"): Fuel("gas", "lpg"),
    ("Térmica", "BioGas"): Fuel("gas", "biogas"),
    ("Térmica", "Diésel"): Fuel("oil", "diesel"),
    ("Térmica", "Fuel Oil"): Fuel("oil", "fuel_oil"),
    ("Térmica", "PetCoke"): Fuel("oil", "petcoke"),  # petroleum coke
    ("Térmica", "Biomasa"): Fuel("biomass"),
    ("Térmica", "Termosolar"): Fuel("solar", "csp"),
}
# Manual map: CEN's combination w/o fuel → each plant's own STAC fuel, by its global ID
FUELS_PER_EGE_TO_STAC: dict[RawFuel, dict[str, Fuels]] = {
    ("Térmica", "Cogeneración"): {  # combined heat and power
        "cen-397": Fuel("wasteheat", "chp"),  # PLANTA DE ACIDO SULFURICO MEJILLONES
        "cen-1844": Fuel("biomass", "chp"),  # ARAUCO MAPA (pulp mill)
        "cen-1907": Fuel("biomass", "chp"),  # NUEVA ALDEA (pulp mill, black liquor)
    },
}

# ======= STATUSES =======
# Map: CEN's status → STAC status
STATUSES_TO_STAC = {
    # square bracket term in CEN's 'central' (name) = the true status, before 'estado'
    "NO_MOSTRAR": "decommissioned",
    "No_Mostrar": "decommissioned",
    "EN_REVISION": "under_construction",
    "En_Revision": "under_construction",
    # the CEN registry's 'estado' (status)
    "Operativa": "active",
    "Operativa – Autodespacho DS88": "active",  # self-dispatched small plant (DS 88)
    "En Pruebas": "active",  # generating, in commissioning tests
    "En Construcción": "under_construction",
    "Fuera de Servicio": "inactive",  # out of service (the registry has no "retired")
}
