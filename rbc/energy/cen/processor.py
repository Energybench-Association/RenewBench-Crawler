"""CEN DATA PROCESSOR."""

from dataclasses import replace
from pathlib import Path

import icechunk as ic
import pandas as pd
from loguru import logger
from rapidfuzz import fuzz

from rbc.config.schema import PublishConfig
from rbc.coordinates.load import Run, load_runs
from rbc.coordinates.map import build_map
from rbc.coordinates.mappings import (
    OFFICIAL_LOCATOR,
    OPERATOR_METADATA,
    SYSOP_CODE_COL,
    SYSOP_NAME_COL,
)
from rbc.coordinates.orchestrator import find_coordinates
from rbc.coordinates.pipelines import build_shared_resources
from rbc.coordinates.utils.tokenizer import strip_square_brackets
from rbc.energy.cen.mappings import (
    COGENERATION,
    STAC_COGENERATION_FUELS,
    STAC_FUELS,
    STAC_NAME_STATUSES,
    STAC_STATUSES,
)
from rbc.energy.cen.registry import CenCoordinateRegistry
from rbc.energy.processing import (
    LOCATION_COLUMNS,
    STAC_COLUMNS,
    ZARR_COLUMNS,
    check_totals_kept,
    codes_as_text,
    combine_fuels,
    ege_ids,
    entity_ids,
    geometries,
    log_unjoined_ids,
    to_layout,
    to_utc,
)
from rbc.energy.utils import load_df_from_file, write_df_to_csv
from rbc.publish.icechunk_store import (
    GenerationStore,
    StorageInfo,
    open_repository,
    repo_prefix,
    resolution,
)
from rbc.publish.stac import (
    DatasetInfo,
    build_collection,
    build_items,
    publish,
    validate,
    write_stac,
)

SOURCE = "cen"

META = OPERATOR_METADATA[SOURCE]
COUNTRY = META["country"]
NAME_COL = META["name_col"]
CODE_COL = META["code_col"]

# Chile's time zone (CEN reports local time, incl. its DST changes)
TZ = "America/Santiago"

# A CEN plant's ('central') operating configuration, reported in rows of its own
CONFIG_COL = "id_opreal"

# CEN API docs v2.0 (2023), p. 57: "La Generación Real del Sistema Eléctrico Nacional,
# corresponde a la Potencia bruta real del sistema, medida en MW" (s. data_sources.md)
MEASUREMENT_TYPE = "gross_output"

# Map: relevant registry columns for STAC (API names) → neutral names for processor
REGISTRY_RENAMES = {
    "id_central": "id",
    "central": "name",
    "estado": "status",
}
# Map: CEN's raw columns → the shared layout's (s. docs/data_sources.md, renamed on loading)
CEN_COLUMNS = {
    "fecha_hora": "timestamp",
    "gen_real_mw": "value",
    "propietario": "operator",
    "potencia_maxima": "generation_capacity",
    "tipo_tecnologia": "fuel_type",
    "subtipo_tecnologia": "fuel_subtype",
}

# What CEN's Collections say about the source, its terms of use and our processing
CEN_TERMS_URL = "https://www.coordinador.cl/terminos-y-condiciones/"
CEN_DATASET = DatasetInfo(
    dataset=SOURCE,
    title="CEN (Chile) hourly generation per plant",
    description=(
        "Hourly gross generation in MW of each power plant of Chile's National "
        "Electric System (SEN), as reported by the Coordinador Eléctrico Nacional "
        "(CEN) through its public information system (SIP) API.\n\n"
        "For academic, non-commercial use only, citing the source, as CEN's terms of "
        f"use allow ({CEN_TERMS_URL}). Not endorsed by CEN.\n\n"
    ),
    license="proprietary",
    license_url=CEN_TERMS_URL,
    providers=[
        {
            "name": "Coordinador Eléctrico Nacional (CEN)",
            "roles": ["producer", "licensor"],
            "url": "https://www.coordinador.cl",
        },
        {
            "name": "Energybench Association",
            "roles": ["processor", "host"],
            "url": "https://energybench.org",
        },
    ],
    citation=(
        "Coordinador Eléctrico Nacional (CEN), Chile: real generation per plant "
        "(Generación Real), Sistema de Información Pública, https://www.coordinador.cl. "
        "Processed and published by the Energybench Association, "
        "https://energybench.org."
    ),
    values="Hourly gross generation (MW)",
    keywords=["Chile", "CEN", "hourly"],
)


def local_day_in_utc(day: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Give a local calendar day's start and end in UTC (23, 24 or 25 hours apart).

    Args:
        day (str): The local date, e.g. '2023-04-01'.

    Returns:
        tuple[pd.Timestamp, pd.Timestamp]: The day's first and the next day's first
            instant in UTC. A midnight a DST change skips starts the day an hour later.
    """
    start, end = (
        pd.Timestamp(d).tz_localize(TZ, nonexistent="shift_forward").tz_convert("UTC")
        for d in (day, pd.Timestamp(day) + pd.Timedelta(days=1))
    )
    return start, end


class CenProcessor:
    """CEN data processor.

    Attributes:
        input_dir (Path): The input directory containing subfolders of raw csv files.
        resources_dir (Path): The directory containing resources for coordinate finding.
        output_dir (Path): The output directory for processed energy generation data.
        coordinates_dir (Path): The output directory for coordinate finding results.
        publish_cfg (PublishConfig): Where and as which release the data is published.
        upload (bool): Whether to write to S3 and the STAC API, or only locally.
        df_registry (pd.DataFrame): The dataframe of EGE info from CEN API registry.
        shared (SharedResources): The dataclass of shared resources for coordinate finding.
    """

    def __init__(
        self,
        token: str,
        input_dir: Path,
        resources_dir: Path,
        output_dir: Path,
        publish_cfg: PublishConfig,
        upload: bool = True,
    ):
        """Initialize the CEN raw data processor.

        Args:
            token (str): The CEN API token for getting the coordinate registry.
            input_dir (Path): The directory containing subfolders of CEN raw data.
            resources_dir (Path): The directory containing resources for coordinate finding.
            output_dir (Path): The output directory for processed energy generation data.
            publish_cfg (PublishConfig): Where and as which release to publish.
            upload (bool, optional): Whether to write the repository to S3 and the STAC
                records to the STAC API. If False, the repository is written into
                `output_dir` and the STAC records only as JSON. Defaults to True.

        Raises:
            ValueError: If publishing while a URL is still the config's placeholder.
        """
        self.input_dir = input_dir
        self.resources_dir = resources_dir
        self.output_dir = output_dir
        self.coordinates_dir = Path(output_dir, "coordinates")
        self.publish_cfg = publish_cfg
        self.upload = upload
        if upload and (unset := publish_cfg.unset_urls()):
            raise ValueError(
                f"Publishing needs real values for {unset} (pass them with -o "
                f"key=value or a local config), or run with --no-publish."
            )

        # FOR COORDINATE FINDING
        # load EGE coordinate information from CEN registry API endpoint
        reg = CenCoordinateRegistry(
            token=token, cache_dir=Path(self.resources_dir, "cen_portal")
        )
        self.df_registry = reg.df_registry.rename(columns=REGISTRY_RENAMES)
        # load dataclass of shared resources for coordinate finding branch
        self.shared = build_shared_resources(
            sysop=SOURCE, resources_dir=self.resources_dir
        )

    # ------------------------------------------------------------------
    # ENTRYPOINT
    # ------------------------------------------------------------------
    def process_data(self) -> None:
        """Process previously downloaded, raw CEN energy generation data.

        Processing includes the following steps:
        1. Load all raw generation data from each CSV dir (e.g. each temporal res) into
           its own EGE dataframe.
        2. Perform coordinate finding to get one EGE dataframe of unique EGEs & their coords.
        3. Convert the EGE dataframe into zarr, and publish to Icechunk repositories.
        4. Generate the STAC Collection and Items (one per EGE), validate them, and
           publish to the STAC API.

        Any inconsistency (e.g. an hour lost in UTC conversion, an unmapped fuel type)
        raises and stops processing, rather than being skipped.
        """
        runs = list(load_runs([self.input_dir], sysop=SOURCE))
        if not runs:
            raise ValueError(f"No raw '{SOURCE}' data found in {self.input_dir}.")

        for run in runs:
            logger.info(f"Processing run '{run.name}'")
            try:
                self._process_run(run)
            except Exception as e:
                logger.exception(f"Processing {run.map_label} failed: {e}")
                raise

    def _process_run(self, run: Run) -> None:
        """Process one run (raw CSV dir) end to end (s. `process_data`).

        Args:
            run (Run): The run's EGE rows and raw files.

        Raises:
            ValueError: If coordinate finding returns no EGEs.
        """
        release = self.publish_cfg.release

        # Step 1.1: CEN-specific coordinate finding
        df_ege = self._get_coordinates(df_eg=run.df)
        # Step 1.2: General coordinate finding for all EGEs that still lack one
        df_ege = find_coordinates(
            replace(run, df=df_ege), shared=self.shared, output_dir=self.coordinates_dir
        )
        if df_ege is None or len(df_ege) == 0:
            raise ValueError(f"Coordinate finding returned no EGEs for {run.name}.")
        build_map([df_ege], labels=[run.map_label], output_dir=self.coordinates_dir)
        entities = sorted(
            ege_ids(SOURCE, df_ege[SYSOP_CODE_COL], df_ege[SYSOP_NAME_COL])
            .dropna()
            .unique()
        )

        # Step 2: generation data into the release's repository, one raw CSV at a time
        storage = self._storage(repo_prefix(SOURCE, run.name))
        step, step_iso = resolution(run.name)  # the raw data dir's, e.g. "1h"
        days = [path.stem for path in run.files]
        store = GenerationStore(
            repo=self._repository(storage),
            tag=release,
            entities=entities,
            start=local_day_in_utc(days[0])[0],
            end=local_day_in_utc(days[-1])[1],
            step=step,
            chunks=(
                self.publish_cfg.chunks.entities,
                int(pd.Timedelta(days=self.publish_cfg.chunks.days) / step),
            ),
        )
        df_eg_unique = pd.DataFrame()  # each EGE's own values, collected from every CSV
        placeholders: list[pd.DataFrame] = []
        zarr_ids: set[str] = set()  # to check both outputs describe the same EGEs
        for path in run.files:
            df_eg = load_df_from_file(path).drop_duplicates()  # remove repeated rows
            df_eg = df_eg.rename(columns=CEN_COLUMNS)  # layout names now!
            df_eg["id"] = ege_ids(SOURCE, df_eg[CODE_COL], df_eg[NAME_COL])
            eg_unique = df_eg[
                ["id", CODE_COL, CONFIG_COL, "operator", "generation_capacity"]
                + ["fuel_type", "fuel_subtype"]
            ]
            # distinct values only, the latest last (files are in date order)
            df_eg_unique = pd.concat([df_eg_unique, eg_unique]).drop_duplicates(
                keep="last"
            )

            df_zarr, df_placeholders = self._prepare_for_zarr(df_eg, day=path.stem)
            store.add(df_zarr["id"], df_zarr["timestamp"], df_zarr["value"])
            zarr_ids.update(df_zarr["id"].dropna())
            placeholders.append(df_placeholders)
        snapshot = store.finish(message=f"{SOURCE} {release}: {days[0]} to {days[-1]}")
        if self.upload and not store.reused:
            logger.warning(
                f"Release '{release}' wrote new objects: readers reach them through "
                f"{storage.href} only once a new batch of temporary links is delivered "
                f"to reception (by the LSDF)."
            )
        # kept for review: CEN's rows for hours the clocks skipped (no real hour)
        write_df_to_csv(
            pd.concat(placeholders, ignore_index=True),
            Path(self.output_dir, f"skipped_hour_placeholders_{run.name}.csv"),
        )

        # Step 3: STAC records, one Item per EGE
        df_stac = self._prepare_for_stac(df_ege, df_eg_unique)
        log_unjoined_ids(zarr_ids, df_stac["id"])
        items = build_items(
            df_stac, store.spans(), CEN_DATASET, release, storage, snapshot
        )
        collection = build_collection(
            CEN_DATASET,
            release,
            items,
            storage,
            snapshot,
            entities=store.entities,
            times=store.times,
            step=step_iso,
        )
        validate(collection, items)
        write_stac(Path(self.output_dir, "stac"), collection, items)
        if self.upload:
            publish(self.publish_cfg.stac_api, collection, items)

    # ------------------------------------------------------------------
    # MAIN HELPERS
    # ------------------------------------------------------------------
    def _get_coordinates(self, df_eg: pd.DataFrame) -> pd.DataFrame:
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
            df_eg (pd.DataFrame): The distinct EGE rows of a run (s. `Run.df`).

        Returns:
            df_ege (pd.DataFrame): The EGE rows, with the 'official.*' columns added.
        """
        # 1. look up all EGE IDs in the registry (the name only for the check in step 2)
        off = OFFICIAL_LOCATOR
        df_ege = self._join_registry(
            df_eg, code_col=CODE_COL, cols=["name", "lat", "lon", "region"]
        )
        df_ege[f"{off}.match_method"] = (
            df_ege[f"{off}.lat"].notna().map({True: f"{off}_id_exact"})
        )

        codes = pd.to_numeric(df_ege[CODE_COL], errors="coerce")
        found = df_ege[f"{off}.id"].notna()
        not_found = sorted(df_ege.loc[codes.notna() & ~found, NAME_COL].unique())
        if not_found:
            logger.warning(f"{len(not_found)} EGEs not in CEN registry: {not_found}")
        if codeless := int(df_ege.loc[codes.isna(), NAME_COL].nunique()):
            logger.info(f"{codeless} EGEs have no numeric '{CODE_COL}' to look up.")

        # 2. sanity check: same ID should mean same name (excluding square bracket terms)
        named = found & df_ege[NAME_COL].notna()
        ratios = pd.Series(
            [
                fuzz.ratio(strip_square_brackets(a) or a, strip_square_brackets(b) or b)
                for a, b in zip(
                    df_ege.loc[named, NAME_COL],
                    df_ege.loc[named, f"{off}.name"],
                )
            ],
            index=df_ege.index[named],
            dtype=float,
        )
        # doubt the ID match only if none of its name variants is alike enough
        best = ratios.groupby(codes[named]).transform("max")
        doubtful = best.index[
            best < 90.0
        ]  # must be alike (allow tiny errors 'PVF'->'PFV')
        for idx in doubtful:
            logger.warning(
                f"ID {df_ege.at[idx, CODE_COL]}: CEN name "
                f"'{df_ege.at[idx, NAME_COL]}' differs from registry name "
                f"'{df_ege.at[idx, f'{off}.name']}' (ratio {ratios[idx]:.0f})"
                f" - left to the pipeline to match."
            )
        doubtful_ids = codes.isin(codes[doubtful])  # all its rows, named or not
        df_ege.loc[
            doubtful_ids, [f"{off}.lat", f"{off}.lon", f"{off}.match_method"]
        ] = None

        # 3. sanity check: the registry point should lie in the registry's own region
        off_region = pd.Index([])
        if (region_reg := self.shared.region_reg) is not None:
            located = df_ege[f"{off}.lat"].notna()
            levels = pd.Series(
                [
                    region_reg.classify_match(COUNTRY, region, (lat, lon))
                    for region, lat, lon in zip(
                        df_ege.loc[located, f"{off}.region"],
                        df_ege.loc[located, f"{off}.lat"],
                        df_ege.loc[located, f"{off}.lon"],
                    )
                ],
                index=df_ege.index[located],
                dtype=object,
            )
            off_region = levels.index[levels == "mismatch"]  # not "within_bounds"
            for code, name, region in (
                df_ege.loc[off_region, [CODE_COL, NAME_COL, f"{off}.region"]]
                .drop_duplicates(subset=CODE_COL)
                .itertuples(index=False)
            ):
                logger.warning(
                    f"ID {code}: registry point of '{name}' lies outside its own region "
                    f"'{region}' - left to the pipeline to match."
                )
            df_ege.loc[
                codes.isin(codes[off_region]),
                [f"{off}.lat", f"{off}.lon", f"{off}.match_method"],
            ] = None

        has_coords = df_ege[f"{off}.lat"].notna()
        logger.info(
            f"Found coordinates for {codes[has_coords].nunique()} of {codes.nunique()} EGE "
            f"IDs (left to the pipeline: {codes[doubtful].nunique()} as names differ, "
            f"{codes[off_region].nunique()} as outside their own region)."
        )

        return df_ege.drop(columns=[f"{off}.id", f"{off}.name"])  # served their purpose

    def _prepare_for_zarr(
        self, df_eg: pd.DataFrame, day: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            tuple[pd.DataFrame, pd.DataFrame]: The generation data in the ZARR_COLUMNS
                layout (UTC tz-aware), and CEN's placeholder rows for an hour the clocks
                skipped.

        Raises:
            ValueError: If an hour can't be placed in UTC exactly once, or if any EGE's
                total changed on the way.
        """
        utc, placeholder = self._hours_to_utc(df_eg, day)
        placeholders = df_eg[placeholder]
        if len(placeholders):
            logger.info(
                f"{day}: CEN's placeholder for the hour the clocks skipped kept out of "
                f"the data ({len(placeholders)} rows, "
                f"{int((placeholders['value'] != 0).sum())} non-zero, "
                f"{placeholders['value'].sum():.1f} MW in total)."
            )
        df_eg, utc = df_eg[~placeholder], utc[~placeholder]

        hours = pd.DataFrame({"c": df_eg[CONFIG_COL].values, "t": utc.values})
        if (twice := hours.duplicated()).any():
            raise ValueError(
                f"{day}: {int(twice.sum())} configuration hours fall on the same UTC "
                f"hour, e.g. {hours[twice].head(3).to_dict('records')}."
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
        return to_layout(summed, ZARR_COLUMNS), placeholders

    @staticmethod
    def _hours_to_utc(df_eg: pd.DataFrame, day: str) -> tuple[pd.Series, pd.Series]:
        """Place one raw CSV's hours in UTC, as CEN reports them around DST changes.

        Args:
            df_eg (pd.DataFrame): A raw CSV's rows ("timestamp" = 'fecha_hora', "hora").
            day (str): The local date the CSV holds, e.g. '2023-04-02'.

        Returns:
            tuple[pd.Series, pd.Series]: Each row's hour in UTC (tz-aware; NaT for a
                placeholder), and which rows are placeholders for a skipped hour.

        Raises:
            ValueError: If 'hora' lies outside 1-25 or disagrees with 'fecha_hora', or a
                row of a DST change appears on a day without that change.
        """
        hora = df_eg["hora"].astype(int)
        if not hora.between(1, 25).all():
            raise ValueError(f"{day}: 'hora' outside 1-25: {sorted(set(hora))}")
        clock = pd.Timestamp(day) + pd.to_timedelta((hora - 1) % 24, unit="h")
        if (pd.to_datetime(df_eg["timestamp"]) != clock).any():
            raise ValueError(f"{day}: 'fecha_hora' disagrees with 'hora'.")

        start, end = local_day_in_utc(day)
        repeated = hora == 25
        if repeated.any():
            prev_start, prev_end = local_day_in_utc(
                str((pd.Timestamp(day) - pd.Timedelta(days=1)).date())
            )
            if prev_end - prev_start != pd.Timedelta(hours=25):
                raise ValueError(
                    f"{day}: 'hora' 25 without a clock change the day before."
                )

        # the only ambiguous clock hour left is Saturday's 23:00, its first one (DST)
        utc = to_utc(
            clock, tz=TZ, dst=pd.Series(True, index=clock.index), skipped="NaT"
        )
        utc[repeated] = start - pd.Timedelta(
            hours=1
        )  # the repeated 23:00, standard time
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

    def _prepare_for_stac(
        self, df_ege: pd.DataFrame, df_eg_unique: pd.DataFrame
    ) -> pd.DataFrame:
        """Prep unique EGE data from coordinate finding (_ege) & raw (_eg_unique) for STAC.

        Return a df with columns STAC_COLUMNS:
        - "id": get from global helper
        - "entity_ids": use "id_central" column value (for `"cen": ...`) and, if found via
           coordinate finding, the ID of the locator used for matching (e.g. `"gem": ...`)
        - "entity_name": use "central" column values from df_eg_unique
        - "entity_type": hard-coded as "facility"
        - "facility_name": -
        - "group_name": -
        - "operator": use "propietario" column values from df_eg_unique
        - "fuel_type", "fuel_subtype", "hybrid_fuel_types": "tipo_tecnologia" and
          "subtipo_tecnologia" mapped to STAC's vocabulary (s. STAC_FUEL_MAPPING)
        - "generation_capacity": use "potencia_maxima" column values from df_eg_unique
        - "storage_power": the same, instead, for batteries
        - "status": the square bracket term in "central", else the registry's "estado",
          mapped to STAC's vocabulary (s. STAC_NAME_STATUSES, STAC_STATUSES)
        - "measurement_type": hard-coded as "gross_output" (s. MEASUREMENT_TYPE)
        - "lat", "lon", "region": from df_ege
        - "geometry": a point, else the region's/country's polygon

        CEN reports a physical EGE ('central'). Sometimes, these are split into different
        operating configurations ('id_opreal') of units subdivided into fuel modes. E.g.
            e.g. "TER KELAR" has 15 operating conditions such as
            "KELAR TG1 (GNL)", "KELAR TG1 (Diesel)", "KELAR TG1 + 0.5 TV (Gas Arg)",
            "KELAR TG1 + TG2 + TV (Gas Arg)", ... = 3 units (TG1, TG2, TV) mixed into >4 conf
        For this reason, the metadata is defined for all provided EGEs as aforementioned.
        Operator, fuels and capacity are the latest the generation data reports. An EGE
        with several generating configurations has no row
        holding its own capacity (each holds one unit and fuel mode's), so its capacity
        is unknown; a battery's charging configuration doesn't count, as it is the same
        unit. A square bracket term in CEN's name (e.g. "[NO_MOSTRAR]") is the EGE's true
        status, else the registry's "estado".

        Args:
            df_ege (pd.DataFrame): The EGEs as coordinate finding returned them.
            df_eg_unique (pd.DataFrame): The EGE information from raw generation data.

        Returns:
            pd.DataFrame: The EGEs in the STAC_COLUMNS layout.
        """
        latest = df_eg_unique.drop_duplicates(subset=["id", CONFIG_COL], keep="last")
        per_ege = latest.groupby("id").agg(operator=("operator", "last"))
        per_ege = per_ege.join(combine_fuels(self._stac_fuels(latest)))
        # a battery charges and discharges through one unit: only generating configurations
        # count, and with several of them no row holds the EGE's own capacity
        generating = latest[latest["fuel_subtype"] != "Retiro"].groupby("id")
        capacity = (
            generating["generation_capacity"]
            .last()
            .where(generating[CONFIG_COL].nunique() == 1)
        )
        storage = per_ege["fuel_type"] == "storage"
        per_ege["generation_capacity"] = capacity.where(
            ~storage.reindex(capacity.index)
        )
        per_ege["storage_power"] = capacity.where(storage.reindex(capacity.index))

        df_ege = self._join_registry(df_ege, code_col=SYSOP_CODE_COL, cols=["status"])
        names = df_ege[SYSOP_NAME_COL]
        codes = df_ege[SYSOP_CODE_COL]
        df_ege = (
            df_ege.assign(
                id=ege_ids(SOURCE, codes, names),
                entity_ids=entity_ids(df_ege, own={"cen": codes_as_text(codes)}),
                entity_name=names.map(
                    lambda n: (
                        (strip_square_brackets(n) or n) if isinstance(n, str) else n
                    )
                ),
                entity_type="facility",  # a CEN 'central' (s. docs/data_sources.md)
                status=self._stac_status(
                    names.str.extract(r"\[([^\]]+)\]", expand=False),
                    df_ege[f"{OFFICIAL_LOCATOR}.status"],
                ),
                measurement_type=MEASUREMENT_TYPE,
            )
            .drop_duplicates(subset="id")  # an EGE in several resolutions
            .merge(per_ege.reset_index(), on="id", how="left")
        )

        df_ege = to_layout(df_ege, STAC_COLUMNS, renames=LOCATION_COLUMNS)
        df_ege["geometry"] = geometries(
            df_ege, country=COUNTRY, region_reg=self.shared.region_reg
        )
        return df_ege

    # ------------------------------------------------------------------
    # SECONDARY HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _stac_fuels(df_eg_unique: pd.DataFrame) -> pd.DataFrame:
        """Map each EGE's reported technology (type, subtype) to STAC's fuel (s. STAC_FUELS).

        Args:
            df_eg_unique (pd.DataFrame): The EGE rows with "id", "id_central", "fuel_type"
                and "fuel_subtype" as CEN reports them.

        Returns:
            pd.DataFrame: One row per EGE and reported combination, with "id",
                "fuel_type" and "fuel_subtype" in STAC's vocabulary.

        Raises:
            ValueError: If a combination has no mapping, listing all such combinations.
        """
        rows, unknown = [], set()
        cols = ["id", CODE_COL, "fuel_type", "fuel_subtype"]
        for ege, code, main, sub in (
            df_eg_unique[cols].drop_duplicates().itertuples(index=False)
        ):
            pair = (main, sub if isinstance(sub, str) else None)
            if pair == COGENERATION:
                fuel = (
                    STAC_COGENERATION_FUELS.get(int(code)) if pd.notna(code) else None
                )
                rows.append((ege, *(fuel or (None, None))))
            elif pair in STAC_FUELS:
                rows.append((ege, *STAC_FUELS[pair]))
            else:
                unknown.add(pair)

        if unknown:
            raise ValueError(
                f"CEN technology types without a STAC fuel: {sorted(unknown, key=str)}"
            )
        return pd.DataFrame(rows, columns=["id", "fuel_type", "fuel_subtype"])

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
        return from_name.fillna(from_registry)

    def _storage(self, prefix: str) -> StorageInfo:
        """Return storage location - on S3 if uploading, else locally.

        Args:
            prefix (str): The repository's prefix, e.g. "energy/cen/PT1H/repository".

        Returns:
            StorageInfo: The repository's location.
        """
        if self.upload:
            cfg = self.publish_cfg.storage
            return StorageInfo(
                prefix=prefix,
                bucket=cfg.bucket,
                data_base_url=self.publish_cfg.data_base_url,
            )
        return StorageInfo(prefix=prefix, local_dir=Path(self.output_dir, "icechunk"))

    def _repository(self, storage: StorageInfo) -> ic.Repository:
        """Open the repository a StorageInfo names.

        Args:
            storage (StorageInfo): The repository's location.

        Returns:
            ic.Repository: The opened repository.
        """
        cfg = self.publish_cfg.storage
        return open_repository(
            storage.prefix,
            local_dir=storage.local_dir,
            endpoint=cfg.endpoint,
            bucket=cfg.bucket,
            s3_region=cfg.s3_region,
        )

    def _join_registry(
        self, df: pd.DataFrame, code_col: str, cols: list[str]
    ) -> pd.DataFrame:
        """Join the given registry columns onto the rows by EGE ID, as 'official.<col>'.

        Args:
            df (pd.DataFrame): Rows of EGEs, with their CEN ID in `code_col`.
            code_col (str): Name of the column holding the CEN ID (e.g. 28.0 or '28').
            cols (list[str]): Registry columns to join (s. `REGISTRY_RENAMES`).

        Returns:
            pd.DataFrame: The rows with "official.id" and the 'official.<col>' columns added.
        """
        key = pd.to_numeric(df[code_col], errors="coerce").astype("Int64")  # '28.0'=28
        reg = self.df_registry[["id", *cols]].rename(
            columns=lambda c: f"{OFFICIAL_LOCATOR}.{c}"
        )
        # left join, so EGEs not in the registry (or without an ID) keep empty values.
        return (
            df.assign(_key=key)  # the code column itself is left as is
            .merge(reg, left_on="_key", right_on=f"{OFFICIAL_LOCATOR}.id", how="left")
            .drop(columns="_key")
        )
