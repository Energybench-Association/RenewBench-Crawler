"""ONS DATA PROCESSOR."""

from pathlib import Path

import icechunk as ic
import pandas as pd
from loguru import logger

from rbc.config.schema import PublishConfig
from rbc.coordinates.load import Run, load_runs
from rbc.coordinates.map import build_map
from rbc.coordinates.mappings import OPERATOR_METADATA, SYSOP_CODE_COL, SYSOP_NAME_COL
from rbc.coordinates.orchestrator import find_coordinates
from rbc.coordinates.pipelines import build_shared_resources
from rbc.coordinates.utils.values import normalize_name
from rbc.energy.ons.mappings import (
    FORECAST_MODES,
    GROUP_MODES,
    PLANT_MODES,
    STAC_FUELS,
    STAC_HYBRID_FUELS,
    UNKNOWN_MEASUREMENT_MODES,
)
from rbc.energy.processing import (
    LOCATION_COLUMNS,
    STAC_COLUMNS,
    ZARR_COLUMNS,
    codes_as_text,
    combine_fuels,
    ege_ids,
    entity_ids,
    geometries,
    log_unjoined_ids,
    to_layout,
    to_utc,
)
from rbc.energy.stac_entity import EntityTypeDefiner
from rbc.energy.utils import load_df_from_file, write_df_to_csv
from rbc.publish.icechunk_store import (
    EnergyIcechunkWriter,
    StorageInfo,
    open_repository,
    repo_prefix,
    resolution,
)
from rbc.publish.stac import (
    DatasetInfo,
    build_collection,
    build_items,
    drop_stac,
    publish_stac,
    validate,
    write_stac,
)

SOURCE = "ons"

META = OPERATOR_METADATA[SOURCE]
COUNTRY = META["country"]
NAME_COL = META["name_col"]
CODE_COL = META["code_col"]

# Brasília time (ONS reports the local clock, incl. Brazil's DST until 2019)
TZ = "America/Sao_Paulo"

# How ONS operates an EGE, e.g. 'TIPO I' or 'Conjunto de Usinas' (a group of plants)
MODE_COL = "cod_modalidadeoperacao"

# ANEEL's plant code ('CEG'), e.g. 'UHE.PH.AM.000190-2.01' (ONS writes '-' or 'ND' if none)
CEG_COL = "ceg"
CEG_PATTERN = r"[A-Z]{3}\.[A-Z]{2}\.[A-Z]{2}\.\d{6}-\d\.\d{2}"

# ONS Submódulo 6.5: "montante de geração de energia ativa bruta, em MWh/h, verificada,
# por usina" (s. data_sources.md)
MEASUREMENT_TYPE = "gross_output"
# ONS dataset description: the Tipo III groups' data "são referentes a previsões de
# geração", taken for all Tipo III plants (s. data_sources.md)
FORECAST_MEASUREMENT_TYPE = "forecast"
# No ONS source says how it gets these values (s. UNKNOWN_MEASUREMENT_MODES)
UNKNOWN_MEASUREMENT_TYPE = "unknown"

# Map: ONS's raw columns → the shared layout's (s. docs/data_sources.md, renamed on loading)
ONS_COLUMNS = {
    "din_instante": "timestamp",
    "val_geracao": "value",
    "nom_tipousina": "fuel_type",
    "nom_tipocombustivel": "fuel_subtype",
}

# What ONS's Collections say about the source, its license and our processing
ONS_DATA_URL = "https://dados.ons.org.br/dataset/geracao-usina-2"
ONS_DATASET = DatasetInfo(
    dataset=SOURCE,
    title="ONS (Brazil) {tres} generation per plant",
    description=(
        "{Tres} generation in MW (the hour's average, MWmed) of each power plant, plant "
        "group ('Conjunto de Usinas') and group of small plants ('Pequenas Usinas') of "
        "Brazil's National Interconnected System (SIN), as published by the Operador "
        "Nacional do Sistema Elétrico (ONS) on its open data portal. Tipo III plants and "
        "their groups hold ONS's generation forecasts, not measurements.\n\n"
        "Licensed under CC-BY (version not stated), crediting ONS as the source "
        f"({ONS_DATA_URL}). Not endorsed by ONS.\n\n"
        "Changes made: ONS's values ('val_geracao') are kept as reported, missing ones "
        "stay missing. Times are converted from Brasília time (incl. DST until 2019) to "
        "UTC; the hour the clocks repeated, which ONS reports in one row for two hours, "
        "is left out. Plant and fuel types, operating modes and IDs are mapped to the "
        "STAC energy extension's fields. Locations are found by matching plant names to "
        "open plant databases (GEM, osm-powerplants, OpenStreetMap).\n\n"
    ),
    license="CC-BY",  # as ONS's notice names it: not an SPDX ID, which needs a version
    license_url="http://opendefinition.org/od/2.1/pt-br/",
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
    values="{Tres} generation (MW)",
    system_operator="Operador Nacional do Sistema Elétrico (ONS)",
    countries={"BR": "Brazil"},
    continents=["South America"],
    keywords=["Brazil", "ONS"],
)


def local_month_in_utc(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Give a local calendar month's start and end in UTC.

    Args:
        month (str): The local month, e.g. '2018-11'.

    Returns:
        tuple[pd.Timestamp, pd.Timestamp]: The month's first and the next month's first
            instant in UTC. A midnight a DST change skips starts it an hour later.
    """
    first = pd.Timestamp(month)
    start, end = (
        d.tz_localize(TZ, nonexistent="shift_forward").tz_convert("UTC")
        for d in (first, first + pd.offsets.MonthBegin(1))
    )
    return start, end


class OnsProcessor:
    """ONS data processor.

    Attributes:
        input_dir (Path): The input directory containing subfolders of raw csv files.
        resources_dir (Path): The directory containing resources for coordinate finding.
        output_dir (Path): The output directory for processed energy generation data.
        coordinates_dir (Path): The output directory for coordinate finding results.
        release (str): The release to write and publish, e.g. "v0.1.0".
        publish_cfg (PublishConfig): Where the data is published.
        publish (bool): Whether to publish the release, or only write it locally.
        periods (list[str] | None): Years or months of raw data to process (None: all).
        shared (SharedResources): The dataclass of shared resources for coordinate finding.
    """

    def __init__(
        self,
        input_dir: Path,
        resources_dir: Path,
        output_dir: Path,
        release: str,
        publish_cfg: PublishConfig,
        publish: bool = True,
        periods: list[str] | None = None,
    ):
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

        Raises:
            ValueError: If publishing while a URL is still the config's placeholder (s.
                `PublishConfig.check_publishable`).
        """
        self.input_dir = input_dir
        self.resources_dir = resources_dir
        self.output_dir = output_dir
        self.coordinates_dir = Path(output_dir, "coordinates")
        self.release = release
        self.publish_cfg = publish_cfg
        self.publish = publish
        self.periods = periods
        if publish:
            publish_cfg.check_publishable()

        # FOR COORDINATE FINDING (ONS has no registry of its own)
        self.shared = build_shared_resources(
            sysop=SOURCE, resources_dir=self.resources_dir
        )
        # FOR STAC (entity types: s. `_prepare_for_stac`)
        self.entity_types = EntityTypeDefiner(sysop=SOURCE)

    # ------------------------------------------------------------------
    # ENTRYPOINT
    # ------------------------------------------------------------------
    def process_data(self) -> None:
        """Process previously downloaded, raw ONS energy generation data.

        Processing includes the following steps:
        1. Load all raw generation data from each CSV dir (e.g. each temporal res) into
           its own EGE dataframe.
        2. Perform coordinate finding to get one EGE dataframe of unique EGEs & their coords.
        3. Convert the EGE dataframe into zarr, and publish to Icechunk repositories.
        4. Generate the STAC Collection and Items (one per EGE), validate them, and
           publish to the STAC API.

        Any inconsistency (e.g. an hour the clocks skipped, an unmapped fuel type)
        raises and stops processing, rather than being skipped.
        """
        runs = list(load_runs([self.input_dir], sysop=SOURCE, periods=self.periods))
        if not runs:
            for_periods = f" for {self.periods}" if self.periods else ""
            raise ValueError(
                f"No raw '{SOURCE}' data found in {self.input_dir}{for_periods}."
            )

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
        release = self.release

        # Step 1: coordinate finding
        df_ege = find_coordinates(
            run, shared=self.shared, output_dir=self.coordinates_dir
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
        months = [path.stem for path in run.files]
        writer = EnergyIcechunkWriter(
            repo=self._repository(storage),
            tag=release,
            entities=entities,
            start=local_month_in_utc(months[0])[0],
            end=local_month_in_utc(months[-1])[1],
            step=step,
            chunks=(
                self.publish_cfg.chunks.entities,
                int(pd.Timedelta(days=self.publish_cfg.chunks.days) / step),
            ),
        )
        df_eg_unique = pd.DataFrame()  # each EGE's own values, collected from every CSV
        ambiguous: list[pd.DataFrame] = []
        zarr_ids: set[str] = set()  # to check both outputs describe the same EGEs
        totals = pd.Series(dtype=float)  # each EGE's total generation
        active: dict[str, set[str]] = {}  # the months each EGE reports values in
        for path in run.files:
            df_eg = load_df_from_file(path).rename(columns=ONS_COLUMNS)  # layout names!
            self._check_modes(df_eg[MODE_COL], month=path.stem)
            df_eg["id"] = ege_ids(SOURCE, df_eg[CODE_COL], df_eg[NAME_COL])
            eg_unique = df_eg[
                ["id", CODE_COL, CEG_COL, MODE_COL, "fuel_type", "fuel_subtype"]
            ]
            # distinct values only, the latest last (files are in date order)
            df_eg_unique = pd.concat([df_eg_unique, eg_unique]).drop_duplicates(
                keep="last"
            )

            df_zarr, df_ambiguous = self._prepare_for_zarr(df_eg, month=path.stem)
            writer.add(df_zarr["id"], df_zarr["timestamp"], df_zarr["value"])
            zarr_ids.update(df_zarr["id"].dropna())
            totals = totals.add(df_zarr.groupby("id")["value"].sum(), fill_value=0.0)
            for ege in df_zarr.dropna(subset=["value"])["id"].unique():
                active.setdefault(ege, set()).add(path.stem)
            ambiguous.append(df_ambiguous)

        snapshot = writer.finish(
            message=f"{SOURCE} {release}: {months[0]} to {months[-1]}"
        )
        if self.publish and not writer.reused:
            logger.warning(
                f"Release '{release}' wrote new objects: readers reach them through "
                f"{storage.href} only once new temporary links are signed for them."
            )
        # kept for review: ONS's one row for the hour the clocks repeated (two real hours)
        kept_out = pd.concat(ambiguous, ignore_index=True)
        if not kept_out.empty:
            name = f"repeated_hour_rows_{run.name}.csv"
            write_df_to_csv(kept_out, Path(self.output_dir, "review", name))

        # Step 3: STAC records, one Item per EGE
        df_stac = self._prepare_for_stac(
            df_ege, df_eg_unique, pd.Series(active, dtype=object), totals
        )
        log_unjoined_ids(zarr_ids, df_stac["id"])
        items = build_items(
            df_stac, writer.spans(), ONS_DATASET, release, storage, snapshot, step_iso
        )
        collection = build_collection(
            ONS_DATASET,
            release,
            items,
            storage,
            snapshot,
            entities=writer.entities,
            times=writer.times,
            step=step_iso,
        )
        validate(collection, items)
        write_stac(Path(self.output_dir, "stac"), collection, items)
        if self.publish and self.publish_cfg.route == "lsdf":
            drop_stac(self.publish_cfg.lsdf.dst_dir_inbox, collection, items)
        elif self.publish:
            publish_stac(self.publish_cfg.s3.url_stac_api, collection, items)

    # ------------------------------------------------------------------
    # MAIN HELPERS
    # ------------------------------------------------------------------
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
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            tuple[pd.DataFrame, pd.DataFrame]: The generation data in the ZARR_COLUMNS
                layout (UTC tz-aware), and ONS's rows for the hour the clocks repeated.

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
        return to_layout(df_zarr, ZARR_COLUMNS), df_repeated

    @staticmethod
    def _hours_to_utc(timestamps: pd.Series, month: str) -> tuple[pd.Series, pd.Series]:
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
            TZ, ambiguous="NaT", nonexistent="shift_forward"
        ).isna()
        # no hour left is ambiguous, so `dst` is never read
        utc = to_utc(
            local[~repeated], tz=TZ, dst=pd.Series(False, index=local[~repeated].index)
        )
        return utc, repeated

    def _prepare_for_stac(
        self,
        df_ege: pd.DataFrame,
        df_eg_unique: pd.DataFrame,
        active: pd.Series,
        totals: pd.Series,
    ) -> pd.DataFrame:
        """Prep unique EGE data from coordinate finding (_ege) & raw (_eg_unique) for STAC.

        Return a df with columns STAC_COLUMNS:
        - "id": get from global helper
        - "entity_ids": "id_ons" value (for `"ons": ...`), ANEEL's "ceg" where ONS gives
          one, and, if found via coordinate finding, the ID of the locator used for
          matching (e.g. `"gem": ...`)
        - "entity_name": use "nom_usina" column values from df_ege
        - "entity_type": "portfolio" for a group of plants (s. GROUP_MODES) without a
          point (several sites), "group" for a single plant that is part of a facility
          (s. `EntityTypeDefiner.find_groups`, by CEG or by name and location), else
          "facility"
        - "facility_name": the facility such a part belongs to
        - "fuel_type", "fuel_subtype", "hybrid_fuel_types": "nom_tipousina" and
          "nom_tipocombustivel" mapped to STAC's vocabulary (s. STAC_FUELS)
        - "measurement_type": "forecast" for Tipo III plants and their groups (s.
          FORECAST_MODES), "unknown" for MMGD groups (s. UNKNOWN_MEASUREMENT_MODES),
          else "gross_output" (s. MEASUREMENT_TYPE)
        - "lat", "lon", "region": from df_ege
        - "geometry": a point, else the region's/country's polygon
        ONS provides no operator, capacity or status, so these stay empty.

        A CEG can be shared by several EGEs (e.g. Itaipu's 50 Hz and 60 Hz sides).

        Args:
            df_ege (pd.DataFrame): The EGEs as coordinate finding returned them.
            df_eg_unique (pd.DataFrame): The EGE information from raw generation data.
            active (pd.Series): The months each EGE reports values in, by its ID.
            totals (pd.Series): Each EGE's total generation, by its ID.

        Returns:
            pd.DataFrame: The EGEs in the STAC_COLUMNS layout.
        """
        latest = df_eg_unique.drop_duplicates(subset="id", keep="last")
        per_ege = latest.set_index("id")[[CEG_COL, MODE_COL]]
        per_ege = per_ege.join(combine_fuels(self._stac_fuels(latest)))

        ceg = per_ege[CEG_COL].astype("string")
        mode = per_ege[MODE_COL]
        per_ege = per_ege.assign(
            ceg=ceg.where(ceg.str.fullmatch(CEG_PATTERN, na=False)),
            measurement_type=pd.Series(MEASUREMENT_TYPE, index=mode.index)
            .mask(mode.isin(FORECAST_MODES), FORECAST_MEASUREMENT_TYPE)
            .mask(mode.isin(UNKNOWN_MEASUREMENT_MODES), UNKNOWN_MEASUREMENT_TYPE),
        )

        names = df_ege[SYSOP_NAME_COL]
        codes = df_ege[SYSOP_CODE_COL]
        df_ege = (
            df_ege.assign(id=ege_ids(SOURCE, codes, names), entity_name=names)
            .drop_duplicates(subset="id")  # an EGE in several resolutions
            .merge(per_ege.reset_index(), on="id", how="left")
        )
        # a group located to one point is one site, else it spans several
        spread = df_ege[MODE_COL].isin(GROUP_MODES) & df_ege["lat"].isna()
        df_ege["entity_type"] = spread.map({True: "portfolio", False: "facility"})
        plants = df_ege[~df_ege[MODE_COL].isin(GROUP_MODES)].rename(
            columns={CEG_COL: "code"}
        )
        groups = self.entity_types.find_groups(
            plants[["id", "entity_name", "code", "lat", "lon"]], active, totals
        )
        df_ege["entity_type"] = (
            df_ege["id"].map(groups["entity_type"]).fillna(df_ege["entity_type"])
        )
        df_ege["facility_name"] = df_ege["id"].map(groups["facility_name"])
        if len(groups):
            logger.info(
                f"{len(groups)} EGEs are parts of {groups['facility_name'].nunique()} "
                f"facilities: {sorted(groups['facility_name'].unique())}"
            )
        df_ege["entity_ids"] = entity_ids(
            df_ege,
            own={"ons": codes_as_text(df_ege[SYSOP_CODE_COL]), "ceg": df_ege[CEG_COL]},
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
        """Map each EGE's plant type and fuel to STAC's fuels (s. STAC_FUELS).

        Both sides are compared normalized: without case, accents or punctuation (s.
        `normalize_name`). A combination of two technologies gives both of their fuels,
        so the EGE becomes a hybrid (s. STAC_HYBRID_FUELS).

        Args:
            df_eg_unique (pd.DataFrame): The EGE rows with "id", "fuel_type" and
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
        for ege, main, sub in (
            df_eg_unique[cols].drop_duplicates().itertuples(index=False)
        ):
            main = main if isinstance(main, str) else None
            sub = sub if isinstance(sub, str) else None
            pair = (normalize_name(main), normalize_name(sub))
            if pair in fuels:
                rows.extend((ege, *fuel) for fuel in fuels[pair])
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

    def _storage(self, prefix: str) -> StorageInfo:
        """Return storage location - the published one if publishing, else locally.

        Args:
            prefix (str): The repository's prefix, e.g. "energy/ons/PT1H/repository".

        Returns:
            StorageInfo: The repository's location.
        """
        if self.publish:
            return StorageInfo(
                prefix=prefix,
                bucket=self.publish_cfg.bucket,
                url_public=self.publish_cfg.url_public,
            )
        return StorageInfo(prefix=prefix, local_dir=Path(self.output_dir, "icechunk"))

    def _repository(self, storage: StorageInfo) -> ic.Repository:
        """Open the repository a StorageInfo names, on the config's route if published.

        Args:
            storage (StorageInfo): The repository's location.

        Returns:
            ic.Repository: The opened repository.
        """
        cfg = self.publish_cfg
        local_dir = storage.local_dir
        # published on the LSDF route: <dst_dir_data>/<bucket>/<prefix>
        if local_dir is None and cfg.route == "lsdf":
            local_dir = Path(cfg.lsdf.dst_dir_data, cfg.bucket)
        return open_repository(
            storage.prefix,
            local_dir=local_dir,
            endpoint=cfg.s3.url_endpoint,
            bucket=cfg.bucket,
        )
