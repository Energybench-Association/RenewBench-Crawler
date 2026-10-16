"""CEN DATA PROCESSOR."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
from loguru import logger
from rapidfuzz import fuzz

from rbc.coordinates.load import load_runs
from rbc.coordinates.map import build_map
from rbc.coordinates.mappings import OFFICIAL_LOCATOR, OPERATOR_METADATA, SYSOP_CODE_COL
from rbc.coordinates.orchestrator import find_coordinates
from rbc.coordinates.pipelines import build_shared_resources
from rbc.coordinates.utils.tokenizer import strip_square_brackets
from rbc.energy.cen.registry import CenCoordinateRegistry
from rbc.energy.utils import load_df_from_file, write_df_to_csv

SOURCE = "cen"

META = OPERATOR_METADATA[SOURCE]
COUNTRY = META["country"]
NAME_COL = META["name_col"]
CODE_COL = META["code_col"]

# Map: relevant registry columns for STAC (API names) → neutral names for processor
REGISTRY_RENAMES = {
    "id_central": "id",
    "central": "name",
    "estado": "status",
}


class CenProcessor:
    """CEN data processor.

    Attributes:
        input_dir (Path): The input directory containing subfolders of raw csv files.
        resources_dir (Path): The directory containing resources for coordinate finding.
        output_dir (Path): The output directory for processed energy generation data.
        coordinates_dir (Path): The output directory for coordinate finding results.
        df_registry (pd.DataFrame): The dataframe of EGE info from CEN API registry.
        shared (SharedResources): The dataclass of shared resources for coordinate finding.
    """

    def __init__(
        self, token: str, input_dir: Path, resources_dir: Path, output_dir: Path
    ):
        """Initialize the CEN raw data processor.

        Args:
            token (str): The CEN API token for getting the coordinate registry.
            input_dir (Path): The directory containing subfolders of CEN raw data.
            resources_dir (Path): The directory containing resources for coordinate finding.
            output_dir (Path): The output directory for processed energy generation data.
        """
        self.input_dir = input_dir
        self.resources_dir = resources_dir
        self.output_dir = output_dir
        self.coordinates_dir = Path(output_dir, "coordinates")

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

    def process_data(self) -> None:
        """Process previously downloaded, raw CEN energy generation data.

        Processing includes the following steps:
        1. Load all raw generation data from each CSV dir (e.g. each temporal res) into
           its own EG dataframe.
        2. Perform coordinate finding to get one EGE dataframe of unique EGEs & their coords.
        3. Reformat EG dataframe to be Zarr-ready.
        4. Extract information from EGE dataframe to create STAC metadata.
        """
        all_df_ege: list[pd.DataFrame] = []
        map_labels: list[str] = []

        # Step 1: one run per CSV dir (e.g. per temporal res), with its raw EG rows
        for run in load_runs([self.input_dir], sysop=SOURCE):
            try:
                logger.info(f"Processing run '{run.name}'")

                # Step 2.1: CEN-specific coordinate finding
                df_ege = self._get_coordinates(df_eg=run.df)

                # Step 2.2: General coordinate finding for all EGEs that still lack one
                df_ege = find_coordinates(
                    replace(run, df=df_ege),
                    shared=self.shared,
                    output_dir=self.coordinates_dir,
                )
                # Use explicit check to avoid pandas NA boolean ambiguity
                if df_ege is not None and len(df_ege) > 0:
                    all_df_ege.append(df_ege)
                    map_labels.append(run.map_label)

                # todo: Step 3: reformat EG data into Zarrs (atm: one CSV at a time, memory!)
                input_dir = Path(self.input_dir).resolve()  # run.files are resolved
                for path in run.files:
                    df_eg = load_df_from_file(path)
                    # - leave names as they are
                    # - standardize time to same timestamp format
                    # - standardize gen value to same unit (MWh?)
                    write_df_to_csv(
                        df_eg, Path(self.output_dir, path.relative_to(input_dir))
                    )

            except Exception as e:
                logger.exception(
                    f"Error occurred while processing {run.map_label}: {e}. Skipping..."
                )

        if not all_df_ege:
            logger.warning("No EGEs found in any run: nothing to map or describe.")
            return

        # Step 4.1: map unique EGEs
        build_map(all_df_ege, labels=map_labels, output_dir=self.coordinates_dir)

        # todo: Step 4.2: translate & standardize column headers of EGE to convert into STAC
        df_ege = pd.concat(all_df_ege, ignore_index=True)
        df_ege = self._join_registry(df_ege, code_col=SYSOP_CODE_COL, cols=["status"])
        # - map column headers to STAC attributes (energy.entity, energy.status, ...)
        # - store as `stac_data.csv` for reference
        # - transform into STAC meta

        out_ege_path = Path(self.output_dir, "stac_data.csv")
        write_df_to_csv(df_ege, out_ege_path)

    def _get_coordinates(self, df_eg: pd.DataFrame) -> pd.DataFrame:
        """Add the coordinates of CEN's own EGE registry to the distinct EGE rows.

        Coordinate finding includes the following steps:
        1. look up each EGE ID in the CEN EGE coordinate registry
        2. compare the registry name with the CEN name ('central'): where they differ, the
           ID link is doubtful, so the EGE is left to the pipeline to match by its own name
        3. check the registry point lies in the registry's own region: where it doesn't, the
           record contradicts itself (e.g. a mislabelled UTM zone), so it's left out likewise

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
