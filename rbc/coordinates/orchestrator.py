"""Coordinate/location finding orchestration.

Orchestrates coordinate finding for power-generation units via a declarative,
per-operator pipeline (see rbc.coordinates.pipelines):
- ENTSO-E uses its own EIC-code-driven pipeline (EntsoePipeline),
- every other operator uses the "default" pipeline (DefaultPipeline).
This setup may be enhanced in the future depending on the operator-specific needs.

`find_coordinates` is the entry point for callers that already hold an operator's rows
(e.g. its processor); `perform_coordinate_finding` is the one for callers that hold
paths, reading the CSVs and rendering a map over every directory it was given.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from rbc.coordinates.load import Run, load_runs
from rbc.coordinates.map import build_map
from rbc.coordinates.mappings import OPERATOR_METADATA
from rbc.coordinates.pipelines import (
    SharedResources,
    build_shared_resources,
    make_pipeline,
)


def find_coordinates(
    run: Run,
    shared: SharedResources | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Find coordinates for one run of an operator's EGEs, returning one row per EGE.

    The run's rows are expected in the operator's own column names (s. OPERATOR_METADATA);
    deduping them to one row per EGE is the pipeline's first step. Callers processing
    several runs should build `shared` once and pass it in every time, rather than paying
    the construction cost (network/CSV/parquet reads) per call.

    Args:
        run (Run): The operator's raw rows and what identifies them (s. `load_runs`).
        shared (SharedResources, Optional): Pre-built resources to reuse. Defaults to
            None, in which case they are built for this call alone, without local files.
        output_dir (Path | None): Path of the output directory. Defaults to None, in
            which case no output files are written.

    Returns:
        pd.DataFrame: Enriched dataframe, one row per unique EGE.

    Raises:
        ValueError: If the source is unknown or its EGEs can't be matched by name.
    """
    _check_matchable(run.sysop)
    shared = shared or build_shared_resources(sysop=run.sysop, resources_dir=None)

    pipeline = make_pipeline(
        run=run,
        output_dir=output_dir,
        gem_loc=shared.gem_loc,
        ppdb_loc=shared.ppdb_loc,
        osm_loc=shared.osm_loc,
        region_reg=shared.region_reg,
        eic_reg=shared.eic_reg,
    )
    return pipeline.run_pipeline()


def perform_coordinate_finding(
    sysop: str,
    input_dirs: list[Path | str],
    output_dir: Path | None = None,
    resources_dir: Path | None = None,
    update: bool = False,
) -> None:
    """Main entry point for coordinating location finding over raw CSV directories.

    Reads each directory's CSVs, runs `find_coordinates` on them and renders one map
    over all the results.

    Args:
        sysop (str): Name of the energy system operator, e.g. "aemo".
        input_dirs (list[Path | str]): Paths of the directories containing input (".csv")
            files (somewhere).
        output_dir (Path | None): Path of the output directory. Defaults to None.
        resources_dir (Path | None): Directory of the resources shared by all energy
            sources (e.g. GEM trackers in `gem/`, OSM files, EIC directory). Defaults to
            None, in which case no local resource files are read or written.
        update (bool, optional): Whether to re-fetch OSM power plant data from the OSM
            (Overpass Turbo) API. Defaults to False.

    Raises:
        ValueError: If the source is unknown or its EGEs can't be matched by name.
    """
    _check_matchable(sysop)

    # Build the expensive (network/CSV/parquet-backed) resources ONCE and share!
    shared = build_shared_resources(
        sysop=sysop, resources_dir=resources_dir, update=update
    )
    logger.info("Shared resources initialized.")

    dataframes: list[pd.DataFrame] = []
    map_labels: list[str] = []

    for run in load_runs(input_dirs, sysop=sysop):
        logger.info(
            f"Running coordinate finding for '{run.name}'\n------------------"
            "---------------------------------------------------------------"
        )
        try:
            df = find_coordinates(run, shared=shared, output_dir=output_dir)

            # Use explicit check to avoid pandas NA boolean ambiguity
            if df is not None and len(df) > 0:
                dataframes.append(df)
                map_labels.append(run.map_label)

        except Exception as e:
            logger.exception(
                f"Error occurred while analysing {run.map_label}: {e}. Skipping..."
            )

    if dataframes:
        build_map(dataframes, labels=map_labels, output_dir=output_dir)
    else:
        logger.warning("No results found for the given input paths.")


def _check_matchable(source: str) -> None:
    """Check that this source's EGEs can be found by matching their names at all.

    Args:
        source (str): Name of the energy source, e.g. "aemo".

    Raises:
        ValueError: If the source is unknown or its EGEs can't be matched by name.
    """
    if source not in OPERATOR_METADATA:
        raise ValueError(f"Unknown energy source: '{source}'")

    if not OPERATOR_METADATA[source]["name_matchable"]:
        raise ValueError(
            f"'{source}' names don't identify single EGEs, so they can't be name-matched. "
            f"Its coordinates come from the operator itself (s. OPERATOR_METADATA)."
        )
