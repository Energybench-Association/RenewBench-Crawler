"""PROCESSING.

The energy processors' shared base (s. `EnergyProcessor`) and output layout.

Every processor writes its operator's data in the same columns, so that the Zarr and STAC
steps can treat all operators alike:
- Zarr data: one row per energy generating entity (EGE) and timestamp = time series EG data,
- STAC data: one row per EGE, its columns named after the STAC energy extension.

Both identify an EGE by its global ID ('id', s. `EnergyProcessor._get_global_ids`), the
STAC item's ID. A column an operator doesn't provide stays empty, i.e. its value is unknown.
"""

from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from loguru import logger

from rbc.config.schema import PublishConfig
from rbc.coordinates.load import Run, load_runs
from rbc.coordinates.map import build_map
from rbc.coordinates.mappings import (
    OPERATOR_METADATA,
    SYSOP_CODE_COL,
    SYSOP_NAME_COL,
    SYSOP_REGION_COL,
)
from rbc.coordinates.orchestrator import find_coordinates
from rbc.coordinates.pipelines import SharedResources, build_shared_resources
from rbc.coordinates.utils.values import normalize_name
from rbc.energy.utils.stac.entity import entity_ids
from rbc.energy.utils.stac.geometry import geometries
from rbc.energy.utils.temporal import get_period_bounds, get_resolution
from rbc.publish.icechunk_store import (
    EnergyIcechunkWriter,
    get_storage_location,
    open_or_create_repository,
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
from rbc.utils.io import load_df_from_file, write_df_to_csv

# Columns of the Zarr data: one row per EGE and timestamp
ZARR_COLUMNS = ["id", "timestamp", "value"]

# Columns of the STAC data: one row per EGE (all but 'id' from STAC energy extension)
STAC_COLUMNS = [
    "id",
    "entity_ids",
    "entity_name",
    "entity_type",
    "facility_name",
    "group_name",
    "operator",
    "fuel_type",
    "fuel_subtype",
    "hybrid_fuel_types",
    "generation_capacity",
    "storage_power",
    "status",
    "measurement_type",
    "lat",
    "lon",
    "region",
    "geometry",
]

# Map: coordinate finding's output columns → the EGE table's (the same for every operator)
LOCATION_COLUMNS = {SYSOP_REGION_COL: "region"}


# ------------------------------------------------------------------
# BASE PROCESSOR
# ------------------------------------------------------------------
class EnergyProcessor(ABC):
    """Abstract base class of energy processor classes.

    This class coordinates the raw energy data transformation into an Icechunk release and
    STAC metadata Collection and Items.
    Subclasses must all set the class attributes and overwrite the abstract methods
    `_prepare_for_zarr` and `_get_stac_fields`. Other hooks are optional.

    Attributes:
        SYSOP (str): The system operator, e.g. "cen".
        DATASET (DatasetInfo): What the operator's Collections say about the dataset.
        TZ (str): The IANA time zone the raw files' periods are local to.
        COLUMNS (dict[str, str]): Map of raw columns → the shared layout's, renamed on
            loading (e.g. "gen_real_mw" → "value").
        EGE_COLUMNS (list[str]): Each EGE's own columns (layout names) kept from the raw
            files for STAC, e.g. "fuel_type"; the latest values win.
        input_dir (Path): The directory containing subfolders of raw csv files.
        resources_dir (Path): The directory containing resources for coordinate finding.
        output_dir (Path): The output directory for processed energy generation data.
        coordinates_dir (Path): The output directory for coordinate finding results.
        release (str): The release to write and publish, e.g. "v0.1.0".
        publish_cfg (PublishConfig): Where the data is published.
        publish (bool): Whether to publish the release, or only write it locally.
        periods (list[str] | None): Years or months of raw data to process (None: all).
        meta (OperatorInfo): The operator's metadata (s. OPERATOR_METADATA).
        shared (SharedResources | None): Coordinate finding's resources, None without it.
    """

    SYSOP: ClassVar[str]
    DATASET: ClassVar[DatasetInfo]
    TZ: ClassVar[str]
    COLUMNS: ClassVar[dict[str, str]]
    EGE_COLUMNS: ClassVar[list[str]]

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
        """Initialize the processor.

        Args:
            input_dir (Path): The directory containing subfolders of raw data.
            resources_dir (Path): The directory containing resources for coordinate finding.
            output_dir (Path): The output directory for processed energy generation data.
            release (str): The release to write and publish, e.g. "v0.1.0".
            publish_cfg (PublishConfig): Where to publish.
            publish (bool, optional): Whether to publish the repository and the STAC
                records (s. `publish_cfg`). If False, the repository is written into
                `output_dir` and the STAC records only as JSON. Defaults to True.
            periods (list[str] | None, optional): Years or months of raw data to
                process, e.g. ["2024", "2025-03"]. Defaults to None, i.e. all raw files.

        Raises:
            TypeError: If the subclass doesn't set all class attributes.
            ValueError: If publishing while a URL is still the config's placeholder (s.
                `PublishConfig.check_publishable`).
        """
        required = ("SYSOP", "DATASET", "TZ", "COLUMNS", "EGE_COLUMNS")
        if missing := [a for a in required if not hasattr(self, a)]:
            raise TypeError(f"{type(self).__name__} doesn't set {missing}.")

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

        self.meta = OPERATOR_METADATA[self.SYSOP]
        self.run_coordinate_finding_pipeline = self.meta["name_matchable"]
        self.shared: SharedResources | None = None

        if self.run_coordinate_finding_pipeline:
            self.shared = build_shared_resources(
                sysop=self.SYSOP, resources_dir=self.resources_dir
            )

    # ------------------------------------------------------------------
    # ENTRYPOINT
    # ------------------------------------------------------------------
    def process_data(self) -> None:
        """Process previously downloaded, raw energy generation data, run by run.

        Raises:
            ValueError: If there is no raw data (for the identified periods).
        """
        runs = list(load_runs([self.input_dir], sysop=self.SYSOP, periods=self.periods))
        if not runs:
            for_periods = f" for {self.periods}" if self.periods else ""
            raise ValueError(
                f"No raw '{self.SYSOP}' data found in {self.input_dir}{for_periods}."
            )

        for run in runs:
            logger.info(f"Processing run '{run.name}'")
            try:
                self._process_run(run)
            except Exception as e:
                logger.exception(f"Processing {run.map_label} failed: {e}")
                raise

    def _process_run(self, run: Run) -> None:
        """Process one run (raw data dir) end to end (s. class docstring).

        Each run (raw data dir such as one temporal resolution) is processed end to end:
        1. Coordinates:
           a) use sysop's own coords/data (via `_get_own_coordinates`), then
           b) coordinate finding pipeline for rest (if the operator is `name_matchable`).
        2. Generation data: one raw file at a time
           a) load, rename to the shared layout (s. COLUMNS), clean (via `_clean_file`),
           b) bring into the Zarr layout (via `_prepare_for_zarr`) and
           c) write to the release's Icechunk repository.
        3. STAC:
           a) build one Item per EGE (via `_get_stac_fields`)
           b) build one Collection per run,
           c) validate, write and publish.
        Rows kept out of the data are written to `<output_dir>/review/` for review.
        Any inconsistency raises and stops processing, rather than being skipped.

        Args:
            run (Run): The run's EGE rows and raw files.

        Raises:
            ValueError: If coordinate finding returns no EGEs for the run.
        """
        # Step 1: find coordinates and store info in a dataframe (one row per EGE)
        df_ege_locators = self._get_own_coordinates(run.df)
        if self.run_coordinate_finding_pipeline:
            df_ege_locators = find_coordinates(
                replace(run, df=df_ege_locators),
                shared=self.shared,
                output_dir=self.coordinates_dir,
            )

        if df_ege_locators is None or len(df_ege_locators) == 0:
            raise ValueError(f"Coordinate finding returned no EGEs for {run.name}.")

        build_map(
            [df_ege_locators], labels=[run.map_label], output_dir=self.coordinates_dir
        )
        df_ege_locators = df_ege_locators.assign(
            id=self._get_global_ids(
                df_ege_locators[SYSOP_CODE_COL], df_ege_locators[SYSOP_NAME_COL]
            )
        )  # without coordinate finding, this is simply run.df

        # Step 2: get raw EG data and build Icechunk repo for release
        step, step_iso = get_resolution(run.name)  # the raw data dir's, e.g. "1h"
        storage = get_storage_location(
            f"energy/{self.SYSOP}/{step_iso}/repository",
            self.publish_cfg,
            publish=self.publish,
            local_dir=Path(self.output_dir, "icechunk"),
        )
        entities = sorted(df_ege_locators["id"].dropna().unique())
        file_periods = [path.stem for path in run.files]  # a year, month or day each

        writer = EnergyIcechunkWriter(
            repo=open_or_create_repository(storage, self.publish_cfg),
            tag=self.release,
            entities=entities,
            start=get_period_bounds(file_periods[0], self.TZ)[0],
            end=get_period_bounds(file_periods[-1], self.TZ)[1],
            step=step,
            chunks=(
                self.publish_cfg.chunks.entities,
                int(pd.Timedelta(days=self.publish_cfg.chunks.days) / step),
            ),
        )

        df_ege_sysop = pd.DataFrame()  # each EGE's own values, taken from every file
        totals = pd.Series(dtype=float)  # each EGE's total generation
        active_periods: dict[str, set[str]] = {}  # time periods each EGE reports values
        zarr_ids: set[str] = set()  # unique IDs of EGEs in Zarr files
        review: dict[str, list[pd.DataFrame]] = {}  # excluded rows by review file

        # Process one raw file at a time
        for path in run.files:
            period = path.stem

            # Build dataframe of energy generation values with "layout" columns
            df_eg = load_df_from_file(path).rename(columns=self.COLUMNS)
            df_eg["id"] = self._get_global_ids_of_rows(df_eg)
            df_eg = self._clean_file(df_eg, period)
            # Collect the EGEs' info from df_eg (distinct values only, latest last) - for STAC
            df_ege_sysop = pd.concat(
                [df_ege_sysop, df_eg[["id", *self.EGE_COLUMNS]]]
            ).drop_duplicates(keep="last")

            # Build Icechunk (Zarr) dataframe with 3 columns (id, timestamp, value)
            df_zarr, kept_out = self._prepare_for_zarr(df_eg, period)
            writer.add(df_zarr["id"], df_zarr["timestamp"], df_zarr["value"])
            zarr_ids.update(df_zarr["id"].dropna())
            totals = totals.add(df_zarr.groupby("id")["value"].sum(), fill_value=0.0)
            for ege_id in df_zarr.dropna(subset=["value"])["id"].unique():
                active_periods.setdefault(ege_id, set()).add(period)

            for name, rows in kept_out.items():
                review.setdefault(name, []).append(rows)

        snapshot = writer.finish(
            message=f"{self.SYSOP} {self.release}: {file_periods[0]} to {file_periods[-1]}"
        )
        if self.publish and not writer.reused:
            logger.warning(
                f"Release '{self.release}' wrote new objects: readers reach them through "
                f"{storage.href} only once new temporary links are signed for them."
            )
        self._write_review(review, run.name)

        # Step 3: STAC records (one Item per EGE, one Collection for the dataset)
        df_stac = self._prepare_for_stac(
            df_ege_locators,
            df_ege_sysop,
            pd.Series(active_periods, dtype=object),
            totals,
        )
        log_unjoined_ids(zarr_ids, df_stac["id"])
        items = build_items(
            df_stac,
            writer.spans(),
            self.DATASET,
            self.release,
            storage,
            snapshot,
            step_iso,
        )
        collection = build_collection(
            self.DATASET,
            self.release,
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
    # HOOKS (operator-specific)
    # ------------------------------------------------------------------
    def _get_own_coordinates(self, df_ege_sysop: pd.DataFrame) -> pd.DataFrame:
        """Add the operator's own coordinates to its EGE rows (e.g. from a registry).

        Without coordinate finding (not `name_matchable`), the result must have
        its layout: one row per EGE, with the 'sysop.*' columns, "lat" and "lon".

        Args:
            df_ege_sysop (pd.DataFrame): The distinct EGE rows of a run (s. `Run.df`).

        Returns:
            pd.DataFrame: The rows with the operator's coordinates; unchanged by default.
        """
        return df_ege_sysop

    def _clean_file(self, df_eg: pd.DataFrame, period: str) -> pd.DataFrame:
        """Check and clean one raw file's rows before they are brought into the layout.

        Args:
            df_eg (pd.DataFrame): A raw file's rows, in layout names, with their "id".
            period (str): The local period the file holds, e.g. '2025-07'.

        Returns:
            pd.DataFrame: The rows to process; unchanged by default.
        """
        return df_eg

    @abstractmethod
    def _prepare_for_zarr(
        self, df_eg: pd.DataFrame, period: str
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Bring a raw file's rows into ZARR_COLUMNS layout ("id", UTC "timestamp", "value").

        Args:
            df_eg (pd.DataFrame): A raw file's cleaned rows (s. `_clean_file`).
            period (str): The local period the file holds, e.g. '2025-07'.

        Returns:
            tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
                df_zarr: The generation data, and
                kept_out: the rows kept out of it for review, by review file name (e.g.
                {"repeated_hour_rows": ...}).
        """

    @abstractmethod
    def _get_stac_fields(
        self,
        df_ege_locators: pd.DataFrame,
        df_ege_sysop: pd.DataFrame,
        active_periods: pd.Series,
        totals: pd.Series,
    ) -> pd.DataFrame:
        """Give each EGE's operator-specific STAC fields, e.g. fuels, status, capacity.

        Args:
            df_ege_locators (pd.DataFrame): Dataframe with EGE info from coordinate finding,
                once per "id" with "entity_name".
            df_ege_sysop (pd.DataFrame): Dataframe with EGE info from sysop's raw EG files,
                (s. EGE_COLUMNS).
            active_periods (pd.Series): The periods each EGE reports values in, by its ID.
            totals (pd.Series): Each EGE's total generation, by its ID.

        Returns:
            pd.DataFrame: Indexed by EGE ID, with any STAC_COLUMNS (e.g. "fuel_type");
                "entity_name" replaces the default. Further columns are available to
                `_get_own_ids`.
        """

    def _get_own_ids(self, df_ege_locators: pd.DataFrame) -> dict[str, pd.Series]:
        """Give each EGE's IDs in the operator's own systems (STAC's energy:entity_ids).

        Args:
            df_ege_locators (pd.DataFrame): Dataframe with EGE info from coordinate finding,
                with their fields (s. `_get_stac_fields`).

        Returns:
            dict[str, pd.Series]: Map of ID system → each EGE's ID in it; by default
                the operator's code (e.g. {"cen": ...}).
        """
        return {self.SYSOP: codes_as_text(df_ege_locators[SYSOP_CODE_COL])}

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _get_global_ids_of_rows(self, df_eg: pd.DataFrame) -> pd.Series:
        """Give a raw file's rows their EGE's global ID, by its code or else its name.

        Args:
            df_eg (pd.DataFrame): A raw file's rows, with the operator's name/code columns.

        Returns:
            pd.Series: Each row's EGE ID (s. `_get_global_ids`).
        """
        code_col = self.meta.get("code_col")
        codes = df_eg[code_col] if code_col else pd.Series(pd.NA, index=df_eg.index)
        return self._get_global_ids(codes=codes, names=df_eg[self.meta["name_col"]])

    @classmethod
    def _get_global_ids(cls, codes: pd.Series, names: pd.Series) -> pd.Series:
        """Give EGEs a readable ID unique across operators: '<SYSOP>-<code, else name>'.

        The key (its code as text, else its name) is written in lowercase ASCII with dashes,
        e.g. a CEN EGE with ID 1907 -> "cen-1907", a CEN EGE with only a name "BESS Andes"
        -> "cen-bess-andes". Key creation is reproducible (same key for an EGE in every run),
        so Zarr and STAC are joined correctly and a new EGE won't take an existing one's key.

        Args:
            codes (pd.Series): The EGEs' codes (missing for some).
            names (pd.Series): The EGEs' names.

        Returns:
            pd.Series: The EGEs' global IDs (e.g. "cen-1907"), missing without a key.

        Raises:
            ValueError: If two keys give the same ID (e.g. "BESS Andes", "BESS-ANDES").
        """
        keys = codes_as_text(codes).fillna(names.astype("string"))
        lookup = {
            key: f"{cls.SYSOP}-{normalize_name(str(key)).replace(' ', '-')}"
            for key in keys.dropna().unique()
        }
        seen: dict[str, object] = {}
        for key, gid in lookup.items():
            if gid in seen:
                raise ValueError(
                    f"'{cls.SYSOP}' EGE keys '{seen[gid]}' and '{key}' share the ID "
                    f"'{gid}'."
                )
            seen[gid] = key
        return keys.map(lookup)

    def _prepare_for_stac(
        self,
        df_ege_locators: pd.DataFrame,
        df_ege_sysop: pd.DataFrame,
        active_periods: pd.Series,
        totals: pd.Series,
    ) -> pd.DataFrame:
        """Bring the EGEs into the STAC_COLUMNS layout, with the operator's fields.

        Shared by all operators: "id" and "entity_name" from coordinate finding's output,
        one row per EGE, "entity_ids" (the operator's own, s. `_get_own_ids`, and the
        locator's that located it), "lat", "lon", "region" and "geometry" (a point, else
        the region's/country's polygon). The rest comes from `_get_stac_fields`.

        Args:
            df_ege_locators (pd.DataFrame): The EGEs as coordinate finding returns them, with
                their "id".
            df_ege_sysop (pd.DataFrame): The EGEs' own values from the raw files.
            active_periods (pd.Series): The periods each EGE reports values in, by its ID.
            totals (pd.Series): Each EGE's total generation, by its ID.

        Returns:
            pd.DataFrame: The EGEs in the STAC_COLUMNS layout.
        """
        df_ege_locators = df_ege_locators.assign(
            entity_name=df_ege_locators[SYSOP_NAME_COL]
        ).drop_duplicates(subset="id")  # an EGE in several resolutions

        fields = self._get_stac_fields(
            df_ege_locators=df_ege_locators,
            df_ege_sysop=df_ege_sysop,
            active_periods=active_periods,
            totals=totals,
        )
        df_ege_locators = df_ege_locators.drop(
            columns=[c for c in fields.columns if c in df_ege_locators]
        ).merge(fields, left_on="id", right_index=True, how="left")
        df_ege_locators["entity_ids"] = entity_ids(
            df_ege_locators, own=self._get_own_ids(df_ege_locators)
        )

        df_stac = to_layout(df_ege_locators, STAC_COLUMNS, renames=LOCATION_COLUMNS)
        region_reg = self.shared.region_reg if self.shared else None
        df_stac["geometry"] = geometries(
            df_stac, country=self.meta["country"], region_reg=region_reg
        )
        return df_stac

    def _write_review(self, review: dict[str, list[pd.DataFrame]], run: str) -> None:
        """Write the rows kept out of the data to `<output_dir>/review/<name>_<run>.csv`.

        Args:
            review (dict[str, list[pd.DataFrame]]): The rows, by review file name.
            run (str): The run's name, e.g. "1h".
        """
        for name, parts in review.items():
            rows = pd.concat(parts, ignore_index=True)
            if not rows.empty:  # only files with content
                write_df_to_csv(rows, Path(self.output_dir, "review", f"{name}_{run}"))


# ------------------------------------------------------------------
# GENERAL HELPERS
# ------------------------------------------------------------------
def codes_as_text(codes: pd.Series) -> pd.Series:
    """Write codes as text, numbers however they were read (116, 116.0, '116' -> '116').

    Args:
        codes (pd.Series): The EGEs' IDs (missing for some), e.g. 116.0 or 'AMBA'.

    Returns:
        pd.Series: The IDs as text, missing where an EGE has none.
    """
    num = pd.to_numeric(codes, errors="coerce")
    whole = num.where(num % 1 == 0).astype("Int64").astype("string")
    return whole.fillna(codes.astype("string"))  # e.g. 'AMBA', '48W000000ABC1234'


def log_unjoined_ids(zarr_ids: set[str], stac_ids: pd.Series) -> None:
    """Log the EGE IDs only one of the Zarr and STAC data has, as both must have them all.

    Args:
        zarr_ids (set[str]): The distinct IDs in the Zarr data.
        stac_ids (pd.Series): The IDs in the STAC data.
    """
    stac_ids = set(stac_ids.dropna())
    for unjoined, lacking in (
        (zarr_ids - stac_ids, "STAC"),
        (stac_ids - zarr_ids, "Zarr"),
    ):
        if unjoined:
            logger.error(
                f"{len(unjoined)} EGE IDs have no {lacking} data, e.g. "
                f"{sorted(unjoined)[:5]}"
            )


def to_layout(
    df: pd.DataFrame, layout: list[str], renames: dict[str, str] | None = None
) -> pd.DataFrame:
    """Bring data into a layout's columns, dropping all others.

    Args:
        df (pd.DataFrame): The operator's data.
        layout (list[str]): The layout's columns, in order (e.g. STAC_COLUMNS).
        renames (dict[str, str], optional): Map of a column → layout column, for any not
            named after the layout yet. Defaults to None.

    Returns:
        pd.DataFrame: The data in the layout's columns, empty where not provided.
    """
    return df.rename(columns=renames or {}).reindex(columns=layout)


def check_totals_kept(before: pd.Series, after: pd.Series, what: str) -> None:
    """Raise if a reshaping step changed any EGE's total, so no value was lost or added.

    Args:
        before (pd.Series): Each EGE's total before the step, indexed by its ID.
        after (pd.Series): Each EGE's total after the step, indexed by its ID.
        what (str): The data the totals are of, for the error, e.g. a file name.

    Raises:
        ValueError: If an EGE's totals differ (beyond float rounding).
    """
    before, after = before.align(after, fill_value=0.0)
    changed = ~np.isclose(before.to_numpy(float), after.to_numpy(float))
    if changed.any():
        diff = (after - before)[changed]
        raise ValueError(
            f"Totals changed for {int(changed.sum())} EGEs in {what}: "
            f"{diff.head(5).round(3).to_dict()}"
        )
