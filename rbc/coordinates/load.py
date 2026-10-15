"""Filesystem-side inputs for coordinate finding.

Resolves the raw CSV directory layout into plain values the pipelines can take as
arguments, so that the pipelines themselves never have to read the filesystem.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from rbc.coordinates.mappings import OPERATOR_COLUMNS, OPERATOR_METADATA
from rbc.energy.utils import DownloadTask, load_df_from_file

TRES_PATTERN = DownloadTask._TRES_PATTERN  # temporal res dir names ("1h", "15min")
DATE_PATTERN = DownloadTask._DATE_PATTERN  # data file stems ("2020-01-05")


@dataclass(frozen=True, eq=False)  # eq=False: a dataframe can't be compared or hashed
class Run:
    """One directory's EGE rows of one operator, with what identifies them.

    Built by `load_runs`. Holds only the distinct EGE rows coordinate finding reads, not
    the time series: processing the generation data goes through `files`, one at a time.
    A caller changing the rows before coordinate finding (e.g. to add its own
    coordinates) hands over a copy: `dataclasses.replace(run, df=new_df)`.

    Attributes:
        sysop (str): Name of the system operator, e.g. "entsoe".
        name (str): Name of the run among others of the same operator, used for log
            lines and output file names, e.g. "15min_10YRO-TEL------P".
        zone (str | None): Bidding zone of the rows, only set for multi-country operators.
        files (tuple[Path, ...]): The directory's raw CSV files, sorted.
        df (pd.DataFrame): The distinct rows of the operator's metadata columns (s.
            OPERATOR_METADATA), in its own column names: one per name/code/fuel combo.
    """

    sysop: str
    name: str
    zone: str | None
    files: tuple[Path, ...]
    df: pd.DataFrame

    @property
    def map_label(self) -> str:
        """Define the Run's label as <sysop>_<name> to be using in mapping.

        Returns:
            str: Label of the run on the map e.g. "entsoe_15min_10YRO-TEL------P".
        """
        return f"{self.sysop}_{self.name}"


def load_runs(input_paths: list[Path | str], sysop: str) -> Iterator[Run]:
    """Yield one run per raw CSV directory of the operator found below the given paths.

    EGE names missing from a directory's rows are recovered from its other temporal
    resolutions before the run is handed out. A directory that can't be loaded is
    logged and skipped, so one broken directory doesn't end the whole run.

    Args:
        input_paths (list[Path | str]): Paths to search for raw CSV directories.
        sysop (str): Name of the system operator, e.g. "entsoe".

    Yields:
        Run: The next directory's dataframe incl. other required Run parameters.
    """
    csv_dirs = [d for d in _collect_dirs(input_paths) if sysop in d.parts]
    logger.info(f"Found {len(csv_dirs)} directories of '{sysop}' raw data.")

    for csv_dir in csv_dirs:
        try:
            name, zone = _get_run_name_and_zone(csv_dir)
            files = tuple(sorted(csv_dir.glob("*.csv")))
            df = recover_names(_load_ege_rows(files, sysop), csv_dir, sysop, name)
        except Exception as e:
            logger.exception(
                f"Error occurred while loading {csv_dir}: {e}. Skipping..."
            )
            continue

        yield Run(sysop=sysop, name=name, zone=zone, files=files, df=df)


def _collect_dirs(input_paths: list[Path | str]) -> list[Path]:
    """Resolve one or more input paths into a flat, sorted list of directories.

    The input list can contain:

    - A single folder — a directory that directly contains the CSV files (no subdirectories).
        Treated as one dir.
    - A container folder — a directory whose children contain the CSV files (somewhere).
        All subdirectories are searched and those collected that contain relevant CSVs.
    - Any mix of the two shapes above — each element is resolved independently and the
        results are concatenated.

    Args:
        input_paths (list): A list of paths of interest.

    Returns:
        list[Path]: Deduplicated, sorted list of resolved directories.
    """
    candidates = [Path(p) for p in input_paths]
    directories: set[Path] = set()

    for path in candidates:
        if not path.is_dir():
            logger.warning(f"'{path}' is not a directory — skipping!")
            continue

        for p in path.rglob("*.csv"):
            if DATE_PATTERN.match(p.stem) and any(
                TRES_PATTERN.match(prt) for prt in p.parts
            ):
                directories.add(p.parent.resolve())

    return sorted(directories)


def _load_ege_rows(files: tuple[Path, ...], sysop: str) -> pd.DataFrame:
    """Read the distinct EGE rows of an operator's CSVs, one file at a time.

    Only the columns the operator's metadata names are read, as they are all coordinate
    finding uses. Memory so grows with the number of EGEs, not of timestamps. Keeping
    each combo's first row in file order leaves the pipeline's own dedupe unchanged.

    Args:
        files (tuple[Path, ...]): The operator's raw CSV files, in the order to read them.
        sysop (str): Name of the system operator, e.g. "entsoe".

    Returns:
        pd.DataFrame: The distinct EGE rows, empty if there are no files.
    """
    meta: dict[str, Any] = dict(OPERATOR_METADATA[sysop])
    cols = {meta[attr] for attr in OPERATOR_COLUMNS if meta.get(attr)}

    all_dfs = [
        load_df_from_file(p, usecols=lambda c: c in cols).drop_duplicates()
        for p in files
    ]
    if not all_dfs:
        logger.warning(f"No CSV files found for '{sysop}' in this directory.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True).drop_duplicates(ignore_index=True)


def _get_run_name_and_zone(input_dir: Path) -> tuple[str, str | None]:
    """Name a directory's run and, if it exists, extract the zone from it.

    A directory sitting directly below a temporal resolution is a zone of a multi-country
    operator, so it contributes both parts of the name and the zone itself, e.g.:
    - "entsoe/15min/10YRO-TEL------P" -> ("15min_10YRO-TEL------P", "10YRO-TEL------P")
    - "ons/1h"                        -> ("1h", None)

    Args:
        input_dir (Path): Directory holding an operator's raw CSV files.

    Returns:
        tuple[str, str | None]: The run's label, and its zone if it has one.
    """
    if TRES_PATTERN.match(input_dir.parent.name):
        return f"{input_dir.parent.name}_{input_dir.name}", input_dir.name

    return input_dir.name, None


def _get_other_tres_dirs(input_dir: Path) -> list[Path]:
    """Find the equivalents of the input directory for other temporal resolutions.

    Swap the tres-component of the input path for every other resolution present, e.g.:
    - input_dir = "entsoe/15min/10YRO-TEL------P" -> ["entsoe/1h/10YRO-TEL------P"]
    - input_dir = "aeso/1h" -> ["aeso/5min"]

    Args:
        input_dir (Path): Directory of the data whose other resolutions are wanted.

    Returns:
        list[Path]: Existing directories of this data in other temporal resolutions or
            empty list, if no other resolutions exists.
    """
    parts = input_dir.parts
    tres_idx = next((i for i, p in enumerate(parts) if TRES_PATTERN.match(p)), None)
    if tres_idx is None:
        return []

    other_dirs: list[Path] = []
    for tres_dir in sorted(Path(*parts[:tres_idx]).iterdir()):
        if not tres_dir.is_dir() or tres_dir.name == parts[tres_idx]:
            continue
        if not TRES_PATTERN.match(tres_dir.name):
            continue
        candidate = Path(tres_dir, *parts[tres_idx + 1 :])
        if candidate.is_dir():
            other_dirs.append(candidate)

    return other_dirs


def _find_ege_names_by_code(
    dirs: list[Path],
    name_col: str,
    code_col: str,
    codes: set[str],
    run: str = "",
) -> dict[str, str]:
    """Find missing EGE names using their codes by searching the CSVs of other dirs.

    Operators may omit an EGE's name entirely in one temporal resolution while naming
    it in another. Without a name, no name-based matching can occur. Example:
    - Entsoe RO 1h:     name_col = 'CET_MINT2_CA', code_col = 30WMINTMINT2---1
    - Entsoe RO 15min:  name_col = '', code_col = 30WMINTMINT2---1

    Files are read only while names are still missing. Codes found under two different
    names keep the first one (and are logged).

    Args:
        dirs (list[Path]): Directories to search, in the order they should be read.
        name_col (str): Name of the operator's EGE name column.
        code_col (str): Name of the operator's EGE code column.
        codes (set[str]): EGE codes to find a name for.
        run (str, optional): Run label to prefix log messages with. Defaults to "".

    Returns:
        dict[str, str]: Map of EGE code → name, for every code a name was found for.
    """
    prefix = f"[{run}] " if run else ""
    found: dict[str, str] = {}

    for other_dir in dirs:
        for path in sorted(other_dir.glob("*.csv")):
            if not DATE_PATTERN.match(path.stem):
                continue

            df_other = load_df_from_file(path)
            if not {name_col, code_col}.issubset(df_other.columns):
                continue

            named = df_other[
                df_other[code_col].isin(codes) & df_other[name_col].notna()
            ][[code_col, name_col]].drop_duplicates()

            for code, name in named.itertuples(index=False):
                if found.setdefault(code, name) != name:
                    logger.warning(
                        f"{prefix}'{code}' named both '{found[code]}' and '{name}' in "
                        f"other temporal resolutions. Keeping the first."
                    )

            if len(found) == len(codes):
                return found

    return found


def recover_names(
    df: pd.DataFrame, csv_dir: Path, sysop: str, run: str = ""
) -> pd.DataFrame:
    """Fill in the EGE names missing from a directory's rows from its other resolutions.

    A code ends up unnamed after deduping exactly when none of its rows carries a name,
    so only those codes are looked up. Filling all of their rows gives the same result
    as filling the deduped row, as the pipeline's named-first sort is stable.

    Args:
        df (pd.DataFrame): The operator's EGE rows, read from `csv_dir`.
        csv_dir (Path): Directory the rows were read from.
        sysop (str): Name of the system operator, e.g. "entsoe".
        run (str, optional): Run name, to prefix log messages with. Defaults to "".

    Returns:
        pd.DataFrame: The rows with any recovered names filled in (as is if none were).
    """
    name_col = OPERATOR_METADATA[sysop]["name_col"]
    code_col = OPERATOR_METADATA[sysop].get("code_col")
    if not code_col or not {name_col, code_col}.issubset(df.columns):
        return df

    unnamed = df[name_col].isna()
    codes = set(df.loc[unnamed, code_col].dropna()) - set(
        df.loc[~unnamed, code_col].dropna()
    )
    dirs = _get_other_tres_dirs(csv_dir) if codes else []
    found = (
        _find_ege_names_by_code(dirs, name_col, code_col, codes, run) if dirs else {}
    )
    if not found:
        return df

    df = df.copy()
    fill = unnamed & df[code_col].isin(found)
    df.loc[fill, name_col] = df.loc[fill, code_col].map(found)
    logger.info(
        f"[{run}] Recovered {len(found)} of {len(codes)} missing EGE names from other "
        f"temporal resolutions: {sorted(found.values())}"
    )
    return df
