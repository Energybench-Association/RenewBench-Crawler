"""CEN DATA PROCESSOR."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
from loguru import logger
from rapidfuzz import fuzz

from rbc.coordinates.load import load_runs
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
from rbc.energy.cen.registry import CenCoordinateRegistry
from rbc.energy.processing import (
    LOCATION_COLUMNS,
    STAC_COLUMNS,
    ZARR_COLUMNS,
    codes_as_text,
    ege_ids,
    entity_ids,
    geometry_level,
    log_unjoined_ids,
    to_layout,
    to_utc_iso,
)
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
# Map: CEN's raw columns → the shared layout's (s. docs/data_sources.md, renamed on loading)
CEN_COLUMNS = {
    "fecha_hora": "timestamp",
    "gen_real_mw": "value",
    "propietario": "operator",
    "potencia_maxima": "generation_capacity",
    "tipo_tecnologia": "fuel_type",
    "subtipo_tecnologia": "fuel_subtype",
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

    # ------------------------------------------------------------------
    # ENTRYPOINT
    # ------------------------------------------------------------------
    def process_data(self) -> None:
        """Process previously downloaded, raw CEN energy generation data.

        Processing includes the following steps:
        1. Load all raw generation data from each CSV dir (e.g. each temporal res) into
           its own EG dataframe.
        2. Perform coordinate finding to get one EGE dataframe of unique EGEs & their coords.
        3. Reformat the EG dataframes into the shared layout for zarr data, one raw CSV at a
           time. todo: finish processing and convert df into Zarr.
        4. Combine EG and EGE dataframe data into the shared layout for STAC metadata
           (s. rbc.energy.processing). todo: finish processing and convert df into STAC files.
        """
        all_df_ege: list[pd.DataFrame] = []
        df_eg_unique = (
            pd.DataFrame()
        )  # each EGE's own values, collected from every raw CSV
        map_labels: list[str] = []
        zarr_ids: set[str] = set()  # to check both outputs describe the same EGEs

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

                # Step 3: reformat EG data into shared Zarr (atm: one CSV at a time, memory!)
                input_dir = Path(self.input_dir).resolve()  # run.files are resolved
                for path in run.files:
                    df_eg = load_df_from_file(path).drop_duplicates()  # remove repeated
                    df_eg = df_eg.rename(columns=CEN_COLUMNS)  # layout names now!
                    df_eg["id"] = ege_ids(SOURCE, df_eg[CODE_COL], df_eg[NAME_COL])
                    eg_unique = df_eg[
                        ["id", "id_opreal", "operator", "generation_capacity"]
                        + ["fuel_type", "fuel_subtype"]
                    ]
                    # distinct values only, the latest last (files are in date order)
                    df_eg_unique = pd.concat([df_eg_unique, eg_unique]).drop_duplicates(
                        keep="last"
                    )

                    df_zarr = self._prepare_for_zarr(df_eg)
                    zarr_ids.update(df_zarr["id"].dropna())
                    write_df_to_csv(
                        df_zarr, Path(self.output_dir, path.relative_to(input_dir))
                    )
                    # todo: convert to Zarr

            except Exception as e:
                logger.exception(
                    f"Error occurred while processing {run.map_label}: {e}. Skipping..."
                )

        if not all_df_ege:
            logger.warning("No EGEs found in any run: nothing to map or describe.")
            return

        # Step 4.1: map unique EGEs
        build_map(all_df_ege, labels=map_labels, output_dir=self.coordinates_dir)

        # Step 4.2: reformat EG & EGE data into shared STAC format
        df_ege = pd.concat(all_df_ege, ignore_index=True)
        df_stac = self._prepare_for_stac(df_ege, df_eg_unique)
        write_df_to_csv(df_stac, Path(self.output_dir, "stac_data.csv"))
        log_unjoined_ids(zarr_ids, df_stac["id"])
        # todo: convert to STAC

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

    @staticmethod
    def _prepare_for_zarr(df_eg: pd.DataFrame) -> pd.DataFrame:
        """Prep raw CSV data (df_eg) for Zarr conversion by bringing it to the shared layout.

        Return a df with columns ZARR_COLUMNS ("id", "timestamp", and "value").
        - "id": get from global helper
        - "timestamp": use "fecha_hora" column values, convert to UTC ISO 8601 format
        - "value": use "gen_real_mw" column values, sum up values for each physical EGE.

        CEN reports a physical EGE ('central'). Sometimes, these are split into different
        operating configurations ('llave_opreal').
        Of CEN's over 1000 EGEs, 53 thermal physical EGEs have such configs.
        As these run at the same time, the EGE's generation output is defined as their sum.
        When the EGE is a battery, the charging value is subtracted from the discharging one.

        Args:
            df_eg (pd.DataFrame): A raw CSV's rows, with their EGE's global 'id'.

        Returns:
            pd.DataFrame: The generation data in the ZARR_COLUMNS layout.
        """
        # convert local time with Chile's DST -> UTC ISO8601
        # (daylight saving exists based on solar peak ~13:45 in January, ~12:45 in August)
        # todo: double entry in fall (rewind), none in spring (jump)! How to handle??
        utc = to_utc_iso(df_eg["timestamp"], tz="America/Santiago")

        # net output: charging ('Retiro', batteries only) is reported positive as well
        net = df_eg["value"].where(df_eg["fuel_subtype"] != "Retiro", -df_eg["value"])

        summed = (
            df_eg.assign(value=net, timestamp=utc)
            .groupby(["timestamp", "id"], as_index=False)["value"]
            .sum(min_count=1)  # all values missing stays missing, not 0
        )
        # todo: write to Zarr
        return to_layout(summed, ZARR_COLUMNS)

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
        - "fuel_type": use "tipo_tecnologia" column values from df_eg_unique
        - "fuel_subtype": use "subtipo_tecnologia" column values from df_eg_unique
        - "generation_capacity": use "potencia_maxima" column values from df_eg_unique
        - "status": use "estado" column values from df_ege
        - "measurement_type": hard-coded as "gross_output"
        - "lat": use "lat" from df_ege
        - "lon": use "lon" from df_ege
        - "region": use "region" from df_ege
        - "geometry_level": calculate
        - "location_method": use "match_method" from df_ege

        CEN reports a physical EGE ('central'). Sometimes, these are split into different
        operating configurations ('llave_opreal') of units subdivided into fuel modes. E.g.
            e.g. "TER KELAR" has 15 operating conditions such as
            "KELAR TG1 (GNL)", "KELAR TG1 (Diesel)", "KELAR TG1 + 0.5 TV (Gas Arg)",
            "KELAR TG1 + TG2 + TV (Gas Arg)", ... = 3 units (TG1, TG2, TV) mixed into >4 conf
        For this reason, the metadata is defined for all provided EGEs as aforementioned.
        Operator, fuels and capacity are the latest the generation data reports. An EGE
        with several generating configurations has no row holding its own capacity (each
        holds one unit and fuel mode's), so its capacity is unknown; a battery's charging
        configuration doesn't count, as it is the same unit. A square bracket term in CEN's
        name (e.g. "[NO_MOSTRAR]") is the EGE's true status, else the registry's "estado".

        Args:
            df_ege (pd.DataFrame): The EGEs as coordinate finding returned them.
            df_eg_unique (pd.DataFrame): The EGE information from raw generation data.

        Returns:
            pd.DataFrame: The EGEs in the STAC_COLUMNS layout.
        """
        latest = df_eg_unique.drop_duplicates(subset=["id", "id_opreal"], keep="last")
        per_ege = latest.groupby("id").agg(
            operator=("operator", "last"),
            # todo: map CEN's types to STAC's fuel vocabulary (formatting step)
            fuel_type=("fuel_type", "last"),
            fuel_subtype=(
                "fuel_subtype",
                lambda s: ", ".join(sorted(set(s.dropna()))) or None,
            ),
        )
        # a battery charges and discharges through one unit: only generating configurations
        # count, and with several of them no row holds the EGE's own capacity
        generating = latest[latest["fuel_subtype"] != "Retiro"].groupby("id")
        per_ege["generation_capacity"] = (
            generating["generation_capacity"]
            .last()
            .where(generating["id_opreal"].nunique() == 1)
        )

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
                status=names.str.extract(r"\[([^\]]+)\]", expand=False).fillna(
                    df_ege[f"{OFFICIAL_LOCATOR}.status"]
                ),
                measurement_type="gross_output",  # todo: no proof (s. data_sources.md)
            )
            .drop_duplicates(subset="id")  # an EGE in several resolutions
            .merge(per_ege.reset_index(), on="id", how="left")
        )

        df_ege = to_layout(df_ege, STAC_COLUMNS, renames=LOCATION_COLUMNS)
        df_ege["geometry_level"] = geometry_level(df_ege)
        return df_ege

    # ------------------------------------------------------------------
    # SECONDARY HELPERS
    # ------------------------------------------------------------------
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
