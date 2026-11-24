"""CEN DATA PROCESSOR."""

from pathlib import Path

import pandas as pd
from loguru import logger
from rapidfuzz import fuzz

from rbc.config.schema import PublishConfig
from rbc.coordinates.mappings import (
    OFFICIAL_LOCATOR,
    OPERATOR_METADATA,
    SYSOP_CODE_COL,
    SYSOP_NAME_COL,
)
from rbc.coordinates.utils.tokenizer import strip_square_brackets
from rbc.coordinates.utils.values import normalize_name
from rbc.energy.cen.mappings import (
    COGENERATION,
    STAC_COGENERATION_FUELS,
    STAC_FUELS,
    STAC_NAME_STATUSES,
    STAC_STATUSES,
)
from rbc.energy.cen.registry import CenCoordinateRegistry
from rbc.energy.utils.processing import (
    ZARR_COLUMNS,
    EnergyProcessor,
    check_totals_kept,
    to_layout,
)
from rbc.energy.utils.stac.fuel import combine_fuels
from rbc.energy.utils.temporal import get_period_bounds, to_utc
from rbc.publish.stac import DatasetInfo

SOURCE = "cen"

# CEN columns
META = OPERATOR_METADATA[SOURCE]
COUNTRY = META["country"]
NAME_COL = META["name_col"]
CODE_COL = META["code_col"]
CONFIG_COL = "id_opreal"  # a CEN EGE's ('central') operating configuration

# Map: CEN's raw columns → the shared layout's (s. docs/data_sources.md, renamed on loading)
CEN_COLUMNS = {
    "hora": "hour",
    "fecha_hora": "timestamp",
    "gen_real_mw": "value",
    "propietario": "operator",
    "potencia_maxima": "generation_capacity",
    "tipo_tecnologia": "fuel_type",
    "subtipo_tecnologia": "fuel_subtype",
}

# Map: relevant registry columns for STAC (API names) → neutral names for processor
REGISTRY_RENAMES = {
    "id_central": "id",
    "central": "name",
    "estado": "status",
}

# STAC measurement type (s. data_sources.md for details)
MEASUREMENT_TYPE = "gross_output"

# STAC Collection details for CEN, its license and our processing
CEN_DATASET = DatasetInfo(
    dataset=SOURCE,
    title="CEN (Chile) {tres} generation per plant",
    description=(
        "{Tres} gross generation in MW of each power plant of Chile's National "
        "Electric System (SEN), as reported by the Coordinador Eléctrico Nacional "
        "(CEN) through its public information system (SIP) API.\n\nFor academic, "
        "non-commercial use only, citing the source, as CEN's terms of use allow "
        "(https://www.coordinador.cl/terminos-y-condiciones/). Not endorsed by CEN.\n\n"
    ),
    license="proprietary",
    license_url="https://www.coordinador.cl/terminos-y-condiciones/",
    providers=[
        {
            "name": "Coordinador Eléctrico Nacional (CEN)",
            "roles": ["producer", "licensor"],
            "url": "https://www.coordinador.cl",
        },
        {
            "name": "EnergyBench Association",
            "roles": ["processor", "host"],
            "url": "https://www.energybench.org",
        },
    ],
    citation=(
        "Coordinador Eléctrico Nacional (CEN), Chile: real generation per plant "
        "(Generación Real), Sistema de Información Pública, https://www.coordinador.cl. "
        "Processed and published by the EnergyBench Association, "
        "https://www.energybench.org."
    ),
    lineage=(
        "Processing into Icechunk repositories:\n"
        "- Values: a plant's operating configurations ('id_opreal') are summed per hour, "
        "a battery's charging counted as negative. Rows CEN's files repeat are dropped; "
        "where their values disagree, the plant's hour is left out.\n"
        "- Times: converted from Chile's local time (incl. DST) to UTC, CEN's placeholder "
        "rows for the hour the clocks skipped dropped.\n\n"
        "Processing into STAC metadata:\n"
        "- Mappings: technologies, statuses and IDs are mapped to the STAC energy "
        "extension's fields.\n"
        "- Locations: from CEN's plant registry, else from matching plant names to open "
        "plant databases (GEM, osm-powerplants, OpenStreetMap)."
    ),
    values="{Tres} gross generation (MW)",
    system_operator="Coordinador Eléctrico Nacional (CEN)",
    countries={"CL": "Chile"},
    continents=["South America"],
    keywords=["Chile", "CEN"],
)


class CenProcessor(EnergyProcessor):
    """CEN data processor (s. `EnergyProcessor`).

    Its own coordinates come from CEN's plant registry (s. `_get_own_coordinates`),
    coordinate finding locates the rest.

    Attributes:
        df_registry (pd.DataFrame): The dataframe of EGE info from CEN API registry.
    """

    SYSOP = SOURCE
    DATASET = CEN_DATASET
    TZ = "America/Santiago"  # Chile's time zone (CEN reports local time w DST changes)
    COLUMNS = CEN_COLUMNS
    EGE_COLUMNS = [
        CODE_COL,
        CONFIG_COL,
        "operator",
        "generation_capacity",
        "fuel_type",
        "fuel_subtype",
    ]

    def __init__(
        self,
        token: str,
        input_dir: Path,
        resources_dir: Path,
        output_dir: Path,
        release: str,
        publish_cfg: PublishConfig,
        publish: bool = True,
        periods: list[str] | None = None,
    ):
        """Initialize the CEN raw data processor.

        Args:
            token (str): The CEN API token for getting the coordinate registry.
            input_dir (Path): The directory containing subfolders of CEN raw data.
            resources_dir (Path): The directory containing resources for coordinate finding.
            output_dir (Path): The output directory for processed energy generation data.
            release (str): The release to write and publish, e.g. "v0.1.0".
            publish_cfg (PublishConfig): Where to publish.
            publish (bool, optional): Whether to publish the repository and the STAC
                records (s. `publish_cfg`). If False, the repository is written into
                `output_dir` and the STAC records only as JSON. Defaults to True.
            periods (list[str] | None, optional): Years or months of raw data to
                process, e.g. ["2024", "2025-03"]. Defaults to None, i.e. all raw files.
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
        # load EGE coordinate information from CEN registry API endpoint
        registry = CenCoordinateRegistry(
            token=token, cache_dir=Path(self.resources_dir, "cen_portal")
        )
        self.df_registry = registry.df_registry.rename(columns=REGISTRY_RENAMES)

    # ------------------------------------------------------------------
    # HOOKS (s. EnergyProcessor)
    # ------------------------------------------------------------------
    def _get_own_coordinates(self, df_ege_sysop: pd.DataFrame) -> pd.DataFrame:
        """Add the coordinates of CEN's own EGE registry to the distinct EGE rows.

        Coordinate finding includes the following steps:
        1. look up each EGE ID in the CEN EGE coordinate registry
        2. compare the registry name with the CEN name ('central'): where they differ, the
           ID link is doubtful, so the EGE is left to the pipeline to match by its own name
        3. check the registry point lies in the registry's own region: where it doesn't, the
           record contradicts itself (e.g. a mislabeled UTM zone), so it's left out likewise

        The rows are deliberately not deduped to one per EGE: the pipeline's own dedupe
        needs every variant (e.g. code-less EGEs, every fuel subtype an EGE reports). The
        registry columns are constant per ID, so they survive that dedupe unchanged. EGEs
        without registry coordinates keep empty ones, for the pipeline to fill.

        Args:
            df_ege_sysop (pd.DataFrame): The distinct EGE rows of a run (s. `Run.df`).

        Returns:
            df_ege_locators (pd.DataFrame): The EGE rows, with the 'official.*' columns added.
        """
        # 1. look up all EGE IDs in the official CEN registry (incl name for check in 2.)
        df_ege_locators = self._join_registry(
            df_ege_sysop, code_col=CODE_COL, cols=["name", "lat", "lon", "region"]
        )
        df_ege_locators[f"{OFFICIAL_LOCATOR}.match_method"] = (
            df_ege_locators[f"{OFFICIAL_LOCATOR}.lat"]
            .notna()
            .map({True: f"{OFFICIAL_LOCATOR}_id_exact"})
        )

        codes = pd.to_numeric(df_ege_locators[CODE_COL], errors="coerce")
        found = df_ege_locators[f"{OFFICIAL_LOCATOR}.id"].notna()
        not_found = sorted(
            df_ege_locators.loc[codes.notna() & ~found, NAME_COL].unique()
        )
        if not_found:
            logger.warning(f"{len(not_found)} EGEs not in CEN registry: {not_found}")
        if codeless := int(df_ege_locators.loc[codes.isna(), NAME_COL].nunique()):
            logger.info(f"{codeless} EGEs have no numeric '{CODE_COL}' to look up.")

        # 2. sanity check: same ID should mean same name (excluding square bracket terms)
        named = found & df_ege_locators[NAME_COL].notna()
        ratios = pd.Series(
            [
                fuzz.ratio(strip_square_brackets(a) or a, strip_square_brackets(b) or b)
                for a, b in zip(
                    df_ege_locators.loc[named, NAME_COL],
                    df_ege_locators.loc[named, f"{OFFICIAL_LOCATOR}.name"],
                )
            ],
            index=df_ege_locators.index[named],
            dtype=float,
        )
        # doubt the ID match only if none of its name variants is alike enough
        best = ratios.groupby(codes[named]).transform("max")
        doubtful = best.index[best < 90.0]  # be alike (allow tiny errors 'PVF'->'PFV')
        for idx in doubtful:
            logger.warning(
                f"ID {df_ege_locators.at[idx, CODE_COL]}: CEN name "
                f"'{df_ege_locators.at[idx, NAME_COL]}' differs from registry name "
                f"'{df_ege_locators.at[idx, f'{OFFICIAL_LOCATOR}.name']}' "
                f"(ratio {ratios[idx]:.0f}) - left to the pipeline to match."
            )
        doubtful_ids = codes.isin(codes[doubtful])  # all its rows, named or not
        df_ege_locators.loc[
            doubtful_ids,
            [
                f"{OFFICIAL_LOCATOR}.lat",
                f"{OFFICIAL_LOCATOR}.lon",
                f"{OFFICIAL_LOCATOR}.match_method",
            ],
        ] = None

        # 3. sanity check: the registry point should lie in the registry's own region
        off_region = pd.Index([])
        region_reg = self.shared.region_reg if self.shared else None
        if region_reg is not None:
            located = df_ege_locators[f"{OFFICIAL_LOCATOR}.lat"].notna()
            levels = pd.Series(
                [
                    region_reg.classify_match(COUNTRY, region, (lat, lon))
                    for region, lat, lon in zip(
                        df_ege_locators.loc[located, f"{OFFICIAL_LOCATOR}.region"],
                        df_ege_locators.loc[located, f"{OFFICIAL_LOCATOR}.lat"],
                        df_ege_locators.loc[located, f"{OFFICIAL_LOCATOR}.lon"],
                    )
                ],
                index=df_ege_locators.index[located],
                dtype=object,
            )
            off_region = levels.index[levels == "mismatch"]  # not "within_bounds"
            for code, name, region in (
                df_ege_locators.loc[
                    off_region, [CODE_COL, NAME_COL, f"{OFFICIAL_LOCATOR}.region"]
                ]
                .drop_duplicates(subset=CODE_COL)
                .itertuples(index=False)
            ):
                logger.warning(
                    f"ID {code}: registry point of '{name}' lies outside its own region "
                    f"'{region}' - left to the pipeline to match."
                )
            df_ege_locators.loc[
                codes.isin(codes[off_region]),
                [
                    f"{OFFICIAL_LOCATOR}.lat",
                    f"{OFFICIAL_LOCATOR}.lon",
                    f"{OFFICIAL_LOCATOR}.match_method",
                ],
            ] = None

        has_coords = df_ege_locators[f"{OFFICIAL_LOCATOR}.lat"].notna()
        logger.info(
            f"Found coordinates for {codes[has_coords].nunique()} of {codes.nunique()} EGE "
            f"IDs (left to the pipeline: {codes[doubtful].nunique()} as names differ, "
            f"{codes[off_region].nunique()} as outside their own region)."
        )

        return df_ege_locators.drop(
            columns=[f"{OFFICIAL_LOCATOR}.id", f"{OFFICIAL_LOCATOR}.name"]
        )  # served their purpose

    def _clean_file(self, df_eg: pd.DataFrame, period: str) -> pd.DataFrame:
        """Remove the rows a raw CSV repeats, identical ones and repeated hours.

        Args:
            df_eg (pd.DataFrame): A raw CSV's rows, in layout names, with their "id".
            period (str): The local date the CSV holds, e.g. '2009-12-31'.

        Returns:
            pd.DataFrame: The rows, each configuration's hour once (s.
                `_drop_repeated_hours`).
        """
        return self._drop_repeated_hours(df_eg.drop_duplicates(), day=period)

    @staticmethod
    def _drop_repeated_hours(df_eg: pd.DataFrame, day: str) -> pd.DataFrame:
        """Keep a configuration's first row for an hour CEN's raw CSV holds more than once.

        Some raw CSVs repeat rows: the *-12-31.csv files of several years hold hours 2-24
        a second (or third) time. Those rows mostly have the same generation, but the EGE
        details from later years (e.g. newer names, capacities, owners). Specific logic is
        needed to remove these duplicate rows. If the ID, hour and generation value are
        identical, the first copy of a row is chosen (complete, with its own period's
        details). If the generation values differ, none of the rows is known to be right -
        the hour is deleted for the whole EGE (incl. all of its  configurations), so that
        its total is missing (NaN) rather than wrong.

        Repeats are matched on the configuration and 'hour', not 'fecha_hora': on the day
        the clocks go back, 'hour' 25 shares its label with 'hour' 1 (s. `_hours_to_utc`).

        Args:
            df_eg (pd.DataFrame): A raw CSV's rows, in the shared layout's names, with
                their EGE's global 'id'.
            day (str): The local date the CSV holds, e.g. '2009-12-31'.

        Returns:
            pd.DataFrame: The rows, each configuration's hour once, and none for an
                EGE's hour whose copies disagree.
        """
        key = [CONFIG_COL, "hour"]
        repeated = df_eg.duplicated(subset=key)
        if not repeated.any():
            return df_eg

        first = df_eg[~repeated].set_index(key)["value"]
        again = df_eg[repeated].set_index(key)["value"]
        kept = first.reindex(again.index)
        differ = (kept != again) & ~(kept.isna() & again.isna())

        # the EGEs' hours in doubt, and every row of them (all their configurations)
        in_doubt = df_eg.set_index(key).index.isin(again.index[differ])
        ege_hours = df_eg.loc[in_doubt, ["id", "hour"]].drop_duplicates()
        left_out = pd.MultiIndex.from_frame(df_eg[["id", "hour"]]).isin(
            pd.MultiIndex.from_frame(ege_hours)
        )
        if left_out.any():
            logger.warning(
                f"{day}: copies of {int(differ.sum())} repeated configuration hours "
                f"report different values, so these EGEs' hours are left out (NaN): "
                f"{ege_hours.groupby('id')['hour'].apply(sorted).to_dict()}."
            )

        logger.warning(
            f"{day}: {int(repeated.sum())} repeated configuration hours dropped (other "
            f"plant details), each one's first row kept unless left out."
        )
        return df_eg[~repeated.to_numpy() & ~left_out]

    def _prepare_for_zarr(
        self, df_eg: pd.DataFrame, day: str
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Bring one raw CSV (one local day) into the shared layout for the repository.

        Return a df with columns ZARR_COLUMNS ("id", "timestamp", and "value"):
        - "id": get from global helper
        - "timestamp": the row's hour in UTC, every real hour kept (s. `_hours_to_utc`)
        - "value": "gen_real_mw" summed for each physical EGE (s. below)

        CEN reports a physical EGE ('central'). Sometimes, these are split into different
        operating configurations ('id_opreal'). Of CEN's over 1000 EGEs, 53 thermal
        physical EGEs have such configs. As these run at the same time, the EGE's
        generation output is defined as their sum. When the EGE is a battery, the
        charging value is subtracted from the discharging one.

        Args:
            df_eg (pd.DataFrame): A raw CSV's rows, with their EGE's global 'id'.
            day (str): The local date the CSV holds, e.g. '2023-04-01'.

        Returns:
            tuple[pd.DataFrame, dict[str, pd.DataFrame]]: The generation data in the
                ZARR_COLUMNS layout (UTC tz-aware), and CEN's placeholder rows for an
                hour the clocks skipped, for review ("skipped_hour_placeholders").

        Raises:
            ValueError: If an hour can't be placed in UTC exactly once, or if any EGE's
                total changed on the way.
        """
        utc, placeholder = self._hours_to_utc(df_eg, day)
        df_placeholders = df_eg[placeholder]
        if len(df_placeholders):
            logger.info(
                f"{day}: CEN's placeholder for the hour the clocks skipped kept out of "
                f"the data ({len(df_placeholders)} rows, "
                f"{int((df_placeholders['value'] != 0).sum())} non-zero, "
                f"{df_placeholders['value'].sum():.1f} MW in total)."
            )
        df_eg, utc = df_eg[~placeholder], utc[~placeholder]

        config_hours = pd.DataFrame(
            {CONFIG_COL: df_eg[CONFIG_COL].to_numpy(), "timestamp": utc.to_numpy()}
        )
        if (twice := config_hours.duplicated()).any():
            raise ValueError(
                f"{day}: {int(twice.sum())} configuration hours fall on the same UTC "
                f"hour, e.g. {config_hours[twice].head(3).to_dict('records')}."
            )

        # net output: charging ('Retiro', batteries only) is reported positive as well
        net = df_eg["value"].where(df_eg["fuel_subtype"] != "Retiro", -df_eg["value"])

        summed = (
            df_eg.assign(value=net, timestamp=utc)
            .groupby(["timestamp", "id"], as_index=False)["value"]
            .sum(min_count=1)  # all values missing stays missing, not 0
        )
        check_totals_kept(
            net.groupby(df_eg["id"]).sum(), summed.groupby("id")["value"].sum(), day
        )
        return (
            to_layout(summed, ZARR_COLUMNS),
            {"skipped_hour_placeholders": df_placeholders},
        )

    def _hours_to_utc(
        self, df_eg: pd.DataFrame, day: str
    ) -> tuple[pd.Series, pd.Series]:
        """Place one raw CSV's hours in UTC, as CEN reports them around DST changes.

        Args:
            df_eg (pd.DataFrame): A raw CSV's rows ("timestamp"='fecha_hora', "hour"='hora').
            day (str): The local date the CSV holds, e.g. '2023-04-02'.

        Returns:
            tuple[pd.Series, pd.Series]: Each row's hour in UTC (tz-aware; NaT for a
                placeholder), and which rows are placeholders for a skipped hour.

        Raises:
            ValueError: If 'hour' lies outside 1-25 or disagrees with 'fecha_hora', or a
                row of a DST change appears on a day without that change.
        """
        hour = df_eg["hour"].astype(int)
        if not hour.between(1, 25).all():
            raise ValueError(f"{day}: 'hora' outside 1-25: {sorted(set(hour))}")
        clock = pd.Timestamp(day) + pd.to_timedelta((hour - 1) % 24, unit="h")
        if (pd.to_datetime(df_eg["timestamp"]) != clock).any():
            raise ValueError(f"{day}: 'fecha_hora' disagrees with 'hora'.")

        start, end = get_period_bounds(day, self.TZ)
        repeated = hour == 25
        if repeated.any():
            prev_day = str((pd.Timestamp(day) - pd.Timedelta(days=1)).date())
            prev_start, prev_end = get_period_bounds(prev_day, self.TZ)
            if prev_end - prev_start != pd.Timedelta(hours=25):
                raise ValueError(
                    f"{day}: 'hora' 25 without a clock change the day before."
                )

        # the only ambiguous clock hour left is Saturday's 23:00, its first one (DST)
        utc = to_utc(
            clock, tz=self.TZ, dst=pd.Series(True, index=clock.index), skipped="NaT"
        )
        # the repeated 23:00, standard time
        utc = utc.mask(repeated, start - pd.Timedelta(hours=1))
        placeholder = utc.isna()
        if placeholder.any() and end - start != pd.Timedelta(hours=23):
            raise ValueError(
                f"{day}: hours the clocks skipped on a day without a change."
            )

        if (
            not utc[~placeholder & ~repeated]
            .between(start, end, inclusive="left")
            .all()
        ):
            raise ValueError(f"{day}: hours outside the local day.")
        return utc, placeholder

    def _get_stac_fields(
        self,
        df_ege_locators: pd.DataFrame,
        df_ege_sysop: pd.DataFrame,
        active_periods: pd.Series,
        totals: pd.Series,
    ) -> pd.DataFrame:
        """Give each EGE's CEN-specific STAC fields (the rest: s. `EnergyProcessor`).

        - "entity_name": "central" without its square bracket term
        - "entity_type": hard-coded as "facility"
        - "operator": use "propietario" column values from df_ege_sysop
        - "fuel_type", "fuel_subtype", "hybrid_fuel_types": "tipo_tecnologia" and
          "subtipo_tecnologia" mapped to STAC's vocabulary (s. STAC_FUEL_MAPPING)
        - "generation_capacity": use "potencia_maxima" column values from df_ege_sysop
        - "storage_power": the same, instead, for batteries
        - "status": the square bracket term in "central", else the registry's "estado",
          mapped to STAC's vocabulary (s. STAC_NAME_STATUSES, STAC_STATUSES)
        - "measurement_type": hard-coded as "gross_output" (s. MEASUREMENT_TYPE)

        CEN reports a physical EGE ('central'). Sometimes, these are split into different
        operating configurations ('id_opreal') of units subdivided into fuel modes. E.g.
            "TER KELAR" has 15 operating conditions such as
            "KELAR TG1 (GNL)", "KELAR TG1 (Diesel)", "KELAR TG1 + 0.5 TV (Gas Arg)",
            "KELAR TG1 + TG2 + TV (Gas Arg)", ... = 3 units (TG1, TG2, TV) mixed into >4 conf
        For this reason, the metadata is defined for all provided EGEs as aforementioned.
        Operator, fuels and capacity are the latest the generation data reports. An EGE
        with several generating configurations has no row holding its own capacity (each
        holds one unit and fuel mode's), so its capacity is unknown; a battery's charging
        configuration doesn't count, as it is the same unit. A square bracket term in CEN's
        name (e.g. "[NO_MOSTRAR]") is the EGE's true status, else the registry's "estado".

        Args:
            df_ege_locators (pd.DataFrame): The EGEs as coordinate finding returned them,
                once per "id".
            df_ege_sysop (pd.DataFrame): The EGE information from raw generation data.
            active_periods (pd.Series): The days each EGE reports values in (unused).
            totals (pd.Series): Each EGE's total generation (unused).

        Returns:
            pd.DataFrame: The fields above, indexed by EGE ID.
        """
        latest = df_ege_sysop.drop_duplicates(subset=["id", CONFIG_COL], keep="last")
        fields_sysop = latest.groupby("id").agg(operator=("operator", "last"))
        fields_sysop = fields_sysop.join(combine_fuels(self._stac_fuels(latest)))
        # A battery charges and discharges through one unit: only generating configs counts,
        # and with several of them no row holds the EGE's own capacity
        generating = latest[latest["fuel_subtype"] != "Retiro"].groupby("id")
        capacity = (
            generating["generation_capacity"]
            .last()
            .where(generating[CONFIG_COL].nunique() == 1)
        )
        is_storage = fields_sysop["fuel_type"] == "storage"
        fields_sysop["generation_capacity"] = capacity.where(
            ~is_storage.reindex(capacity.index)
        )
        fields_sysop["storage_power"] = capacity.where(
            is_storage.reindex(capacity.index)
        )

        df_ege_locators = self._join_registry(
            df_ege_locators, code_col=SYSOP_CODE_COL, cols=["status"]
        )
        names = df_ege_locators[SYSOP_NAME_COL]
        fields = pd.DataFrame(
            {
                "entity_name": names.map(
                    lambda n: (
                        (strip_square_brackets(n) or n) if isinstance(n, str) else n
                    )
                ).to_numpy(),
                "entity_type": "facility",  # a CEN 'central' (s. docs/data_sources.md)
                "status": self._stac_status(
                    names.str.extract(r"\[([^\]]+)\]", expand=False),
                    df_ege_locators[f"{OFFICIAL_LOCATOR}.status"],
                ).to_numpy(),
                "measurement_type": MEASUREMENT_TYPE,
            },
            index=pd.Index(df_ege_locators["id"], name="id"),
        )
        fields = fields[~fields.index.duplicated()]  # a registry ID listed twice
        return fields.join(fields_sysop)

    # ------------------------------------------------------------------
    # SECONDARY HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _stac_fuels(df_ege_sysop: pd.DataFrame) -> pd.DataFrame:
        """Map each EGE's reported technology (type, subtype) to STAC's fuel (s. STAC_FUELS).

        CEN spells the same technology in several ways (e.g. 'Cogeneración' and
        'Cogeneracion'), so both sides are compared normalized: without case, accents or
        punctuation (s. `normalize_name`).

        Args:
            df_ege_sysop (pd.DataFrame): The EGE rows with "id", "id_central", "fuel_type"
                and "fuel_subtype" as CEN reports them.

        Returns:
            pd.DataFrame: One row per EGE and reported combination, with "id",
                "fuel_type" and "fuel_subtype" in STAC's vocabulary.

        Raises:
            ValueError: If a combination has no mapping, listing all such combinations.
        """
        fuels = {
            (normalize_name(main), normalize_name(sub)): fuel
            for (main, sub), fuel in STAC_FUELS.items()
        }
        cogeneration = (
            normalize_name(COGENERATION[0]),
            normalize_name(COGENERATION[1]),
        )

        rows, unknown = [], set()
        cols = ["id", CODE_COL, "fuel_type", "fuel_subtype"]
        for ege_id, code, main, sub in (
            df_ege_sysop[cols].drop_duplicates().itertuples(index=False)
        ):
            main = main if isinstance(main, str) else None
            sub = sub if isinstance(sub, str) else None
            pair = (normalize_name(main), normalize_name(sub))
            if pair == cogeneration:
                fuel = (
                    STAC_COGENERATION_FUELS.get(int(code)) if pd.notna(code) else None
                )
                rows.append((ege_id, *(fuel or (None, None))))
            elif pair in fuels:
                rows.append((ege_id, *fuels[pair]))
            else:
                unknown.add((main, sub))  # as CEN spells it

        if unknown:
            raise ValueError(
                f"CEN technology types without a STAC fuel: {sorted(unknown, key=str)}"
            )
        # an EGE reported in two spellings maps to the same fuel twice: keep it once
        return pd.DataFrame(
            rows, columns=["id", "fuel_type", "fuel_subtype"]
        ).drop_duplicates(ignore_index=True)

    @staticmethod
    def _stac_status(name_terms: pd.Series, estado: pd.Series) -> pd.Series:
        """Map each EGE's status to STAC's - its name's square bracket term, else 'estado'.

        Args:
            name_terms (pd.Series): The square bracket term in CEN's name per EGE (e.g.
                "NO_MOSTRAR"; missing if none), s. STAC_NAME_STATUSES.
            estado (pd.Series): The registry's status per EGE (missing if not listed), s.
                STAC_STATUSES.

        Returns:
            pd.Series: The STAC status, missing where neither gives one.

        Raises:
            ValueError: If a term or status has no mapping, listing all such values.
        """
        from_name = name_terms.map(STAC_NAME_STATUSES)
        from_registry = estado.map(STAC_STATUSES)
        unknown = sorted(
            set(name_terms[name_terms.notna() & from_name.isna()])
            | set(estado[estado.notna() & from_registry.isna()])
        )
        if unknown:
            raise ValueError(f"CEN statuses without STAC status: {unknown}")
        return from_name.where(from_name.notna(), from_registry)  # where vs depr fillna

    def _join_registry(
        self, df: pd.DataFrame, code_col: str, cols: list[str]
    ) -> pd.DataFrame:
        """Join the official CEN registry columns to the rows by EGE ID, as 'official.<col>'.

        Args:
            df (pd.DataFrame): Rows of EGEs, with their CEN ID in `code_col`.
            code_col (str): Name of the column holding the CEN ID (e.g. 28.0 or '28').
            cols (list[str]): Registry columns to join (s. `REGISTRY_RENAMES`).

        Returns:
            pd.DataFrame: The rows with "official.id" and the 'official.<col>' columns added.
        """
        key = pd.to_numeric(df[code_col], errors="coerce").astype("Int64")  # '28.0'=28
        registry = self.df_registry[["id", *cols]].rename(
            columns=lambda c: f"{OFFICIAL_LOCATOR}.{c}"
        )
        # left join, so EGEs not in the registry (or without an ID) keep empty values.
        return (
            df.assign(_key=key)  # the code column itself is left as is
            .merge(
                registry, left_on="_key", right_on=f"{OFFICIAL_LOCATOR}.id", how="left"
            )
            .drop(columns="_key")
        )
