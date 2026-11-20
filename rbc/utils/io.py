"""IO.

Reading and writing files.
"""

import json
import zipfile
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger

from rbc.utils.errors import InvalidError


def write_df_to_csv(df: pd.DataFrame, file_path: Path, index: bool = False) -> None:
    """Write dataframe to csv file.

    Args:
        df (pd.DataFrame): Pandas dataframe to be stored in csv file.
        file_path (Path): Path to the csv file.
        index (bool, optional): Whether to include the df index in the csv (True) or not
            (False). Defaults to False.
    """
    file_path = file_path.with_suffix(".csv")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(file_path, index=index)
    logger.info(f"Successfully wrote dataframe to '{file_path}'")


def write_dict_to_json(data: dict, file_path: Path) -> None:
    """Write dictionary to json file.

    Args:
        data (dict): Dictionary to be stored in json file.
        file_path (Path): Path to the json file.
    """
    file_path = file_path.with_suffix(".json")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Successfully wrote dictionary to '{file_path}'")


def load_df_from_file(
    file_path: Path | str, file_type: Literal[".csv", ".xlsx"] | None = None, **args
) -> pd.DataFrame:
    """Load pandas dataframe from a file such as an Excel or csv (file or URL directly).

    Args:
        file_path (Path | str): Path to the Excel file.
        file_type (str | None): Manual file type definition for loading. Defaults to None.
        args (optional): Additional arguments to be passed to pandas loading
            function, i.e. sheet_name for an Excel or index_col for a CSV.

    Returns:
        pd.DataFrame: Pandas dataframe extracted from the provided file.

    Raises:
        InvalidError: If the file doesn't have one of the expected suffixes, if the file
            doesn't exist, if invalid arguments (args) were provided for loading with pandas.
        RETRY_ERRORS: If the file is a URL that is inaccessible for some reason (passed
            through, s. `EnergyDownloader.RETRY_ERRORS`).
    """
    # get suffix as: the defined type OR one of the path suffixes, if it's a valid option
    suffix: str = file_type or next(
        (s for s in Path(file_path).suffixes if s in (".csv", ".xlsx")), ""
    )

    try:
        if suffix == ".xlsx":
            df = pd.read_excel(str(file_path), **args)

        elif suffix == ".csv":
            df = pd.read_csv(str(file_path), **args)

        else:
            raise InvalidError(
                f"Invalid extension in '{file_path}' - not a csv or xlsx!"
            )

    # URL errors (s. EnergyDownloader.RETRY_ERRORS) pass through to the downloader's retries
    except FileNotFoundError:
        raise InvalidError(f"Invalid path - file '{file_path}' does not exist!")

    except TypeError as e:
        raise InvalidError(f"Invalid argument for loading from '{file_path}': {e}")

    logger.info(f"Successfully loaded dataframe from '{file_path}'")
    return df


def load_excel_from_file(file_path: Path | str) -> pd.ExcelFile:
    """Load an Excel file from the provided URL.

    Args:
        file_path (str): URL to download data from.

    Returns:
        pd.ExcelFile: Excel file extracted from the provided URL.

    Raises:
        InvalidError: If the file isn't an xlsx, if the file doesn't exist
        RETRY_ERRORS: If the file is a URL that is inaccessible for some reason (passed
            through, s. `EnergyDownloader.RETRY_ERRORS`).
    """
    if Path(file_path).suffix != ".xlsx":
        raise InvalidError(f"Invalid extension in '{file_path}' - not an xlsx!")

    try:
        xlsx_file = pd.ExcelFile(file_path)

    # URL errors (s. EnergyDownloader.RETRY_ERRORS) pass through to the downloader's retries
    except FileNotFoundError:
        raise InvalidError(f"Invalid path - file '{file_path}' does not exist!")

    except zipfile.BadZipFile:
        raise InvalidError(f"Invalid file content - '{file_path}' is not a valid xlsx!")

    logger.info(f"Successfully loaded excel from '{file_path}'")
    return xlsx_file
