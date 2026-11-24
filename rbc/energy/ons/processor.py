"""ONS DATA PROCESSOR."""

from pathlib import Path

import pandas as pd
from loguru import logger

from rbc.config.schema import PublishConfig
from rbc.coordinates.mappings import OPERATOR_METADATA
from rbc.coordinates.utils.values import normalize_name
from rbc.energy.ons.mappings import (
    FORECAST_MODES,
    GROUP_MODES,
    PLANT_MODES,
    STAC_FUELS,
    STAC_HYBRID_FUELS,
    UNKNOWN_MEASUREMENT_MODES,
)
from rbc.energy.utils.processing import ZARR_COLUMNS, EnergyProcessor, to_layout
from rbc.energy.utils.stac.entity import EntityTypeDefiner
from rbc.energy.utils.stac.fuel import combine_fuels
from rbc.energy.utils.temporal import to_utc
from rbc.publish.stac import DatasetInfo

SOURCE = "ons"

# ONS columns
META = OPERATOR_METADATA[SOURCE]
CODE_COL = META["code_col"]
MODE_COL = (
    "cod_modalidadeoperacao"  # Operation mode, e.g. 'TIPO I'/'Conjunto de Usinas'/
)
CEG_COL = "ceg"  # ANEEL's plant code, e.g. 'UHE.PH.AM.000190-2.01' ('-'/'ND' if none)
CEG_PATTERN = r"[A-Z]{3}\.[A-Z]{2}\.[A-Z]{2}\.\d{6}-\d\.\d{2}"

# Map: ONS's raw columns → the shared layout's (s. docs/data_sources.md, renamed on loading)
ONS_COLUMNS = {
    "din_instante": "timestamp",
    "val_geracao": "value",
    "nom_tipousina": "fuel_type",
    "nom_tipocombustivel": "fuel_subtype",
}

# STAC measurement types (s. data_sources.md for details)
DEFAULT_MEASUREMENT_TYPE = "gross_output"
FORECAST_MEASUREMENT_TYPE = "forecast"  # all type III EGEs and EGE groups
UNKNOWN_MEASUREMENT_TYPE = "unknown"  # all groups with unknown output

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
        entity_types (EntityTypeDefiner): Finds the EGEs that are parts of a facility.
    """

    DATASET = ONS_DATASET
    TZ = "America/Sao_Paulo"  # Brazil time (ONS reports local clock w DST until 2019)
    COLUMNS = ONS_COLUMNS
    EGE_COLUMNS = [CODE_COL, CEG_COL, MODE_COL, "fuel_type", "fuel_subtype"]

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
        self.entity_types = EntityTypeDefiner(sysop=SOURCE)

    # ------------------------------------------------------------------
    # HOOKS (s. EnergyProcessor)
    # ------------------------------------------------------------------
    def _clean_file(self, df_eg: pd.DataFrame, period: str) -> pd.DataFrame:
        """Check that a raw CSV holds only operating modes the processor knows.

        Args:
            df_eg (pd.DataFrame): A raw CSV's rows, in layout names, with their "id".
            period (str): The local month the CSV holds, e.g. '2025-07'.

        Returns:
            pd.DataFrame: The rows, unchanged (s. `_check_modes`).
        """
        self._check_modes(df_eg[MODE_COL], month=period)
        return df_eg

    @staticmethod
    def _check_modes(modes: pd.Series, month: str) -> None:
        """Raise if a raw CSV has an operating mode the processor doesn't know.

        A mode decides an EGE's entity and measurement type (s. GROUP_MODES,
        FORECAST_MODES), so a new one must be classified before it is published.

        Args:
            modes (pd.Series): A raw CSV's 'cod_modalidadeoperacao' values.
            month (str): The local month the CSV holds, e.g. '2025-07'.

        Raises:
            ValueError: If a mode is missing or neither a plant's nor a group's.
        """
        known = PLANT_MODES | GROUP_MODES
        unknown = sorted(set(modes.fillna("<missing>")) - known)
        if unknown:
            raise ValueError(f"{month}: ONS operating modes not classified: {unknown}")

    def _prepare_for_zarr(
        self, df_eg: pd.DataFrame, month: str
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Bring one raw CSV (one local month) into the shared layout for the repository.

        Return a df with columns ZARR_COLUMNS ("id", "timestamp", and "value"):
        - "id": get from global helper
        - "timestamp": "din_instante" in UTC (s. `_hours_to_utc`)
        - "value": "val_geracao" (MWmed: the hour's average in MW), missing ones kept

        ONS reports one row per EGE and hour, so no values are combined.

        Args:
            df_eg (pd.DataFrame): A raw CSV's rows, with their EGE's global 'id'.
            month (str): The local month the CSV holds, e.g. '2018-02'.

        Returns:
            tuple[pd.DataFrame, dict[str, pd.DataFrame]]: The generation data in the
                ZARR_COLUMNS layout (UTC tz-aware), and ONS's rows for the hour the clocks
                repeated, for review ("repeated_hour_rows").

        Raises:
            ValueError: If an EGE's hour is reported twice.
        """
        if (twice := df_eg.duplicated(subset=["id", "timestamp"])).any():
            raise ValueError(
                f"{month}: {int(twice.sum())} EGE hours reported twice, e.g. "
                f"{df_eg.loc[twice, ['id', 'timestamp']].head(3).to_dict('records')}."
            )

        utc, repeated = self._hours_to_utc(df_eg["timestamp"], month)
        df_repeated = df_eg[repeated]
        if len(df_repeated):
            logger.info(
                f"{month}: ONS's one row for the hour the clocks repeated kept out of "
                f"the data, as it can't tell which of the two hours it is "
                f"({len(df_repeated)} rows, at {sorted(set(df_repeated['timestamp']))})."
            )
        df_zarr = df_eg[~repeated].assign(timestamp=utc)
        return to_layout(df_zarr, ZARR_COLUMNS), {"repeated_hour_rows": df_repeated}

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
            ValueError: If a timestamp lies outside the month, off the hour, or in an
                hour the clocks skipped (s. `to_utc`).
        """
        local = pd.to_datetime(timestamps)
        if (local.dt.strftime("%Y-%m") != month).any():
            raise ValueError(f"{month}: 'din_instante' outside the month.")
        if (local != local.dt.floor("h")).any():
            raise ValueError(f"{month}: 'din_instante' not on the hour.")

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

        - "entity_type": "portfolio" for a group of plants (s. GROUP_MODES) without a
          point (several sites), "group" for a single plant that is part of a facility
          (s. `EntityTypeDefiner.find_groups`, by CEG or by name and location), else
          "facility"
        - "facility_name": the facility such a part belongs to
        - "fuel_type", "fuel_subtype", "hybrid_fuel_types": "nom_tipousina" and
          "nom_tipocombustivel" mapped to STAC's vocabulary (s. STAC_FUELS)
        - "measurement_type": "forecast" for Tipo III plants and their groups (s.
          FORECAST_MODES), "unknown" for MMGD groups (s. UNKNOWN_MEASUREMENT_MODES),
          else "gross_output" (s. DEFAULT_MEASUREMENT_TYPE)
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
        latest = df_ege_sysop.drop_duplicates(subset="id", keep="last")
        fields_sysop = latest.set_index("id")[[CEG_COL, MODE_COL]]
        fields_sysop = fields_sysop.join(combine_fuels(self._stac_fuels(latest)))

        ceg = fields_sysop[CEG_COL].astype("string")
        mode = fields_sysop[MODE_COL]
        fields_sysop = fields_sysop.assign(
            ceg=ceg.where(ceg.str.fullmatch(CEG_PATTERN, na=False)),
            measurement_type=pd.Series(DEFAULT_MEASUREMENT_TYPE, index=mode.index)
            .mask(mode.isin(FORECAST_MODES), FORECAST_MEASUREMENT_TYPE)
            .mask(mode.isin(UNKNOWN_MEASUREMENT_MODES), UNKNOWN_MEASUREMENT_TYPE),
        )

        df_ege_locators = df_ege_locators.set_index("id").join(
            fields_sysop[[CEG_COL, MODE_COL]]
        )
        fields = fields_sysop.reindex(df_ege_locators.index)  # also EGEs without values
        # a group located to one point is one site, else it spans several
        spread = (
            df_ege_locators[MODE_COL].isin(GROUP_MODES) & df_ege_locators["lat"].isna()
        )
        fields["entity_type"] = spread.map({True: "portfolio", False: "facility"})
        plants = df_ege_locators[
            ~df_ege_locators[MODE_COL].isin(GROUP_MODES)
        ].reset_index()
        groups = self.entity_types.find_groups(
            plants.rename(columns={CEG_COL: "code"})[
                ["id", "entity_name", "code", "lat", "lon"]
            ],
            active_periods,
            totals,
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

    # ------------------------------------------------------------------
    # SECONDARY HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _stac_fuels(df_ege_sysop: pd.DataFrame) -> pd.DataFrame:
        """Map each EGE's plant type and fuel to STAC's fuels (s. STAC_FUELS).

        Both sides are compared normalized: without case, accents or punctuation (s.
        `normalize_name`). A combination of two technologies gives both of their fuels,
        so the EGE becomes a hybrid (s. STAC_HYBRID_FUELS).

        Args:
            df_ege_sysop (pd.DataFrame): The EGE rows with "id", "fuel_type" and
                "fuel_subtype" as ONS reports them.

        Returns:
            pd.DataFrame: One row per EGE and fuel, with "id", "fuel_type" and
                "fuel_subtype" in STAC's vocabulary.

        Raises:
            ValueError: If a combination has no mapping, listing all such combinations.
        """
        fuels = {
            (normalize_name(main), normalize_name(sub)): [fuel]
            for (main, sub), fuel in STAC_FUELS.items()
        } | {
            (normalize_name(main), normalize_name(sub)): hybrid
            for (main, sub), hybrid in STAC_HYBRID_FUELS.items()
        }

        rows: list[tuple] = []
        unknown: set[tuple] = set()
        cols = ["id", "fuel_type", "fuel_subtype"]
        for ege_id, main, sub in (
            df_ege_sysop[cols].drop_duplicates().itertuples(index=False)
        ):
            main = main if isinstance(main, str) else None
            sub = sub if isinstance(sub, str) else None
            pair = (normalize_name(main), normalize_name(sub))
            if pair in fuels:
                rows.extend((ege_id, *fuel) for fuel in fuels[pair])
            else:
                unknown.add((main, sub))  # as ONS spells it

        if unknown:
            raise ValueError(
                f"ONS plant and fuel types without a STAC fuel: "
                f"{sorted(unknown, key=str)}"
            )
        return pd.DataFrame(
            rows, columns=["id", "fuel_type", "fuel_subtype"]
        ).drop_duplicates(ignore_index=True)
