"""ONS DATA PROCESSOR."""

from pathlib import Path

import pandas as pd
from loguru import logger

from rbc.config.schema import PublishConfig
from rbc.coordinates.mappings import OPERATOR_METADATA
from rbc.energy.ons.mappings import (
    ENTITY_TYPES_TO_STAC,
    FUELS_TO_STAC,
    MEASUREMENT_TYPES_TO_STAC,
)
from rbc.energy.utils.processing import ZARR_COLS, EnergyProcessor
from rbc.energy.utils.stac.entity import EntityTypeDefiner
from rbc.energy.utils.stac.fuel import FuelTypeDefiner
from rbc.energy.utils.stac.measurement import MeasurementTypeDefiner
from rbc.energy.utils.temporal import to_utc
from rbc.publish.stac import DatasetInfo
from rbc.utils.errors import DataStructureError

SOURCE = "ons"

# ONS columns
META = OPERATOR_METADATA[SOURCE]
CODE_COL = META["code_col"]
MODE_COL = "cod_modalidadeoperacao"  # Operation mode, e.g. TIPO I, Conjunto de Usinas
CEG_COL = "ceg"  # ANEEL's plant code, e.g. 'UHE.PH.AM.000190-2.01' ('-'/'ND' if none)
CEG_PATTERN = r"[A-Z]{3}\.[A-Z]{2}\.[A-Z]{2}\.\d{6}-\d\.\d{2}"

# Map: ONS's raw columns → the shared layout's (s. docs/data_sources.md, renamed on loading)
ONS_COLUMNS = {
    "din_instante": "timestamp",
    "val_geracao": "value",
    "nom_tipousina": "fuel_type",
    "nom_tipocombustivel": "fuel_subtype",
}

# STAC Collection details for ONS, its license and our processing
ONS_DATA_URL = "https://dados.ons.org.br/dataset/geracao-usina-2"
ONS_DATASET = DatasetInfo(
    sysop=SOURCE,
    sysop_long="Operador Nacional do Sistema Elétrico (ONS)",
    title="ONS (Brazil) {tres_verbose} generation per plant",
    description=(
        "{Tres_verbose} generation in MW (the hour's average, MWmed) of each power plant, "
        "plant group ('Conjunto de Usinas') and group of small plants ('Pequenas Usinas') of "
        "Brazil's National Interconnected System (SIN), as published by the Operador "
        "Nacional do Sistema Elétrico (ONS) on its open data portal. Tipo III plants and "
        "their groups hold ONS's generation forecasts, not measurements.\n\n"
        "Licensed under CC-BY (version not stated), crediting ONS as the source "
        f"({ONS_DATA_URL}). Not endorsed by ONS.\n\n"
    ),
    license="CC-BY",  # as ONS's notice names it: not an SPDX ID, which needs a version
    license_url="http://opendefinition.org/od/2.1/pt-br/",
    license_notice_url="https://dados.ons.org.br/dataset/geracao-usina-2#:~:text=Aviso,br",
    providers=[
        {
            "name": "Operador Nacional do Sistema Elétrico (ONS)",
            "roles": ["producer", "licensor"],
            "url": "https://www.ons.org.br",
        },
        {
            "name": "EnergyBench Association",
            "roles": ["processor", "host"],
            "url": "https://www.energybench.org",
        },
    ],
    citation=(
        "Operador Nacional do Sistema Elétrico (ONS), Brazil: generation per plant "
        f"(Geração por Usina em Base Horária), {ONS_DATA_URL}. Processed and published "
        "by the EnergyBench Association, https://www.energybench.org."
    ),
    lineage=(
        "Processing into Icechunk repositories:\n"
        "- Values: ONS's values ('val_geracao') are kept as reported, missing ones stay "
        "missing.\n"
        "- Times: converted from Brasília time (incl. DST until 2019) to UTC; the hour the "
        "clocks repeated, which ONS reports in one row for two hours, is left out.\n\n"
        "Processing into STAC metadata:\n"
        "- Mappings: plant and fuel types, operating modes and IDs are mapped to the STAC "
        "energy extension's fields.\n"
        "- Locations: found by matching plant names to open plant databases (GEM, "
        "osm-powerplants, OpenStreetMap)."
    ),
    values="{Tres_verbose} generation (MW)",
    countries={"BR": "Brazil"},
    continents=["South America"],
    keywords=["Brazil", "ONS"],
)


class OnsProcessor(EnergyProcessor):
    """ONS data processor (s. `EnergyProcessor`).

    ONS has no registry of its own: coordinate finding locates all its EGEs.

    Attributes:
        fuel_def (FuelTypeDefiner): Definer for fuel-related energy STAC values.
        entity_def (EntityTypeDefiner): Definer for entity-related energy STAC values.
        measurement_def (MeasurementTypeDefiner): Definer for measurement_type STAC values.
    """

    DATASET = ONS_DATASET
    TZ = "America/Sao_Paulo"  # Brazil time (ONS reports local clock w DST until 2019)
    COLUMNS = ONS_COLUMNS
    EGE_COLUMNS = [CODE_COL, CEG_COL, MODE_COL, "fuel_type", "fuel_subtype"]
    STAC_DEFAULTS = {  # s. docs/data_sources.md)
        "entity_type": "facility"
    }

    def __init__(
        self,
        input_dir: Path,
        resources_dir: Path,
        output_dir: Path,
        release: str,
        publish_cfg: PublishConfig,
        publish: bool = True,
        periods: list[str] | None = None,
    ) -> None:
        """Initialize the ONS raw data processor.

        Args:
            input_dir (Path): The directory containing subfolders of ONS raw data.
            resources_dir (Path): The directory containing resources for coordinate finding.
            output_dir (Path): The output directory for processed energy generation data.
            release (str): The release to write and publish, e.g. "v0.1.0".
            publish_cfg (PublishConfig): Where to publish.
            publish (bool, optional): Whether to publish the repository and the STAC
                records (s. `publish_cfg`). If False, the repository is written into
                `output_dir` and the STAC records only as JSON. Defaults to True.
            periods (list[str] | None, optional): Years or months of raw data to
                process, e.g. ["2018", "2020-03"]. Defaults to None, i.e. all raw files.
        """
        super().__init__(
            input_dir=input_dir,
            resources_dir=resources_dir,
            output_dir=output_dir,
            release=release,
            publish_cfg=publish_cfg,
            publish=publish,
            periods=periods,
        )

        # STAC fields for ONS (s. ons/mappings.py)
        self.fuel_def = FuelTypeDefiner(FUELS_TO_STAC)
        self.entity_def = EntityTypeDefiner(SOURCE, ENTITY_TYPES_TO_STAC)
        self.measurement_def = MeasurementTypeDefiner(MEASUREMENT_TYPES_TO_STAC)

    # ------------------------------------------------------------------
    # HOOKS (s. EnergyProcessor)
    # ------------------------------------------------------------------
    def _clean_file(
        self, df_eg: pd.DataFrame, period: str
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Check that a raw CSV holds only operating modes the processor knows.

        A mode decides an EGE's entity and measurement type (ENTITY_TYPES_TO_STAC,
        MEASUREMENT_TYPES_TO_STAC), so a new one must be classified there before it is used.

        Args:
            df_eg (pd.DataFrame): A raw CSV's rows, in layout names, with their "id".
            period (str): The local month the CSV holds, e.g. '2025-07'.

        Returns:
            tuple[pd.DataFrame, dict[str, pd.DataFrame]]: The rows, unchanged, and no
                rows kept out.

        Raises:
            DataStructureError: If a mode is missing or has no entity or measurement type.
        """
        for definer in (self.entity_def, self.measurement_def):
            definer.map(df_eg[MODE_COL], required=True)  # raises for unknown modes

        return df_eg, {}

    def _prepare_for_zarr(
        self, df_eg: pd.DataFrame, month: str
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Bring one raw CSV (one local month) into the shared layout for the repository.

        Return a df with columns ZARR_COLS ("id", "timestamp", and "value"):
        - "id": get from global helper
        - "timestamp": "din_instante" in UTC (s. `_hours_to_utc`)
        - "value": "val_geracao" (MWmed: the hour's average in MW), missing ones kept

        ONS reports one row per EGE and hour, so no values are combined.

        Args:
            df_eg (pd.DataFrame): A raw CSV's rows, with their EGE's global 'id'.
            month (str): The local month the CSV holds, e.g. '2018-02'.

        Returns:
            tuple[pd.DataFrame, dict[str, pd.DataFrame]]: The generation data in the
                ZARR_COLS layout (UTC tz-aware), and ONS's rows for the hour the clocks
                repeated, for review ("repeated_hour_rows").
        """
        utc, repeated = self._hours_to_utc(df_eg["timestamp"], month)
        df_repeated = df_eg[repeated]
        if len(df_repeated):
            logger.info(
                f"{month}: ONS's one row for the hour the clocks repeated kept out of "
                f"the data, as it can't tell which of the two hours it is "
                f"({len(df_repeated)} rows, at {sorted(set(df_repeated['timestamp']))})."
            )

        # define outputs
        df_zarr = df_eg[~repeated].assign(timestamp=utc).reindex(columns=ZARR_COLS)
        kept_out = {"repeated_hour_rows": df_repeated}
        return (df_zarr, kept_out)

    def _hours_to_utc(
        self, timestamps: pd.Series, month: str
    ) -> tuple[pd.Series, pd.Series]:
        """Place one raw CSV's hours in UTC, as ONS reports them around DST changes.

        ONS reports the local clock hour ('din_instante'). Around DST changes (until 2019):
        - clocks set forward (Sunday 00:00 → 01:00): the skipped 00:00 has no rows.
        - clocks set back (Sunday 00:00 → Saturday 23:00): the repeated 23:00 has one row
          for two real hours, so it is left out (both hours missing).

        Args:
            timestamps (pd.Series): A raw CSV's naive local timestamps ('din_instante').
            month (str): The local month the CSV holds, e.g. '2018-02'.

        Returns:
            tuple[pd.Series, pd.Series]: The other rows' hours in UTC (tz-aware), and
                which rows are in the repeated hour.

        Raises:
            DataStructureError: If a timestamp lies outside the month.
            ValueError: If a timestamp lies in an hour the clocks skipped (s. `to_utc`).
        """
        local = pd.to_datetime(timestamps)
        if (local.dt.strftime("%Y-%m") != month).any():
            raise DataStructureError("'din_instante' outside the month.")

        repeated = local.dt.tz_localize(
            self.TZ, ambiguous="NaT", nonexistent="shift_forward"
        ).isna()
        # no hour left is ambiguous, so `dst` is never read
        utc = to_utc(
            local[~repeated],
            tz=self.TZ,
            dst=pd.Series(False, index=local[~repeated].index),
        )
        return utc, repeated

    def _get_stac_fields(
        self,
        df_ege_locators: pd.DataFrame,
        df_ege_sysop: pd.DataFrame,
        active_periods: pd.Series,
        totals: pd.Series,
    ) -> pd.DataFrame:
        """Give each EGE's ONS-specific STAC fields (the rest: s. `EnergyProcessor`).

        - "entity_type": the mode's (s. ENTITY_TYPES_TO_STAC): "portfolio" for a group
          of plants without a point (several sites), "group" for a single plant that is
          part of a facility (s. `EntityTypeDefiner.find_groups`, by CEG or by name and
          location), else "facility" (s. STAC_DEFAULTS)
        - "facility_name": the facility such a part belongs to
        - "fuel_type", "fuel_subtype", "hybrid_fuel_types": "nom_tipousina" and
          "nom_tipocombustivel" mapped to STAC's vocabulary (s. FUELS_TO_STAC)
        - "measurement_type": the mode's (s. MEASUREMENT_TYPES_TO_STAC), e.g.
          "forecast_output" for Tipo III plants and their groups
        - "ceg": ANEEL's code where ONS gives a valid one (for `_get_own_ids`)
        ONS provides no operator, capacity or status, so these stay empty.

        A CEG can be shared by several EGEs (e.g. Itaipu's 50 Hz and 60 Hz sides).

        Args:
            df_ege_locators (pd.DataFrame): The EGEs as coordinate finding returned them,
                once per "id" with "entity_name".
            df_ege_sysop (pd.DataFrame): The EGE information from raw generation data.
            active_periods (pd.Series): The months each EGE reports values in, by its ID.
            totals (pd.Series): Each EGE's total generation, by its ID.

        Returns:
            pd.DataFrame: The fields above, indexed by EGE ID.
        """
        # Trim dataframe down to the one row per EGE, keeping the newest
        df_ege_sysop = df_ege_sysop.drop_duplicates(subset="id", keep="last")
        fields_sysop = df_ege_sysop.set_index("id")[[CEG_COL, MODE_COL]]

        # --- ADD ENERGY FIELD(S): fuel_type, fuel_subtype, hybrid_fuel_types
        fields_sysop = fields_sysop.join(self.fuel_def.map(df=df_ege_sysop))

        # --- ADD ENERGY FIELD(S): measurement_type
        fields_sysop["measurement_type"] = self.measurement_def.map(
            values=fields_sysop[MODE_COL]
        )

        # Combine relevant content from both df_ege_* dfs into one "fields" df
        ceg = fields_sysop[CEG_COL].astype("string")
        fields_sysop[CEG_COL] = ceg.where(ceg.str.fullmatch(CEG_PATTERN, na=False))
        df_ege = df_ege_locators.set_index("id").join(fields_sysop[[CEG_COL, MODE_COL]])
        fields = fields_sysop.reindex(df_ege.index)  # also EGEs without values

        # --- ADD ENERGY FIELD(S): entity_type, facility_name (where entity_type="group")
        # For logic: see data_sources.md & ons/mappings.py
        by_mode = self.entity_def.map(df_ege[MODE_COL])
        by_mode = by_mode.fillna(self.STAC_DEFAULTS["entity_type"])  # EGEs w/o values
        located_portfolio_mask = (by_mode == "portfolio") & df_ege["lat"].notna()
        fields["entity_type"] = by_mode.mask(located_portfolio_mask, "facility")

        # find "facilities" that are actually "group" (part of same fac) & the facility_name
        facilities = df_ege[by_mode == "facility"].reset_index()
        facilities = facilities.rename(columns={CEG_COL: "code"})
        groups = self.entity_def.find_groups(
            df_ege=facilities[["id", "entity_name", "code", "lat", "lon"]],
            active_periods=active_periods,
            totals=totals,
        )
        ids = fields.index.to_series()
        fields["entity_type"] = ids.map(groups["entity_type"]).fillna(
            fields["entity_type"]
        )
        fields["facility_name"] = ids.map(groups["facility_name"])
        if len(groups):
            logger.info(
                f"{len(groups)} EGEs are parts of {groups['facility_name'].nunique()} "
                f"facilities: {sorted(groups['facility_name'].unique())}"
            )

        return fields

    def _get_own_ids(self, df_ege_locators: pd.DataFrame) -> dict[str, pd.Series]:
        """Give each EGE's ONS ID and, where valid, ANEEL's CEG (energy:entity_ids).

        Args:
            df_ege_locators (pd.DataFrame): The EGEs, with their fields (s. `_get_stac_fields`).

        Returns:
            dict[str, pd.Series]: {"ons": ..., "ceg": ...}.
        """
        return {
            **super()._get_own_ids(df_ege_locators),
            "ceg": df_ege_locators[CEG_COL],
        }
