# tests/utils/test_io.py
"""Tests for reading and writing files."""

import zipfile
from pathlib import Path
from typing import Callable, Literal
from unittest.mock import patch

import pandas as pd
import pytest
from requests import exceptions

from rbc.utils.errors import InvalidError
from rbc.utils.io import (
    load_df_from_file,
    load_excel_from_file,
    write_df_to_csv,
    write_dict_to_json,
)


def test_write_df_to_csv(tmp_path: Path) -> None:
    """Happy path for _write_df_to_csv, ensuring writing a dataframe to a csv works.

    Args:
        tmp_path (Path): Path to temporary directory.
    """
    non_csv_path = Path(tmp_path, "invalid.txt")
    csv_path = Path(tmp_path, "invalid.csv")
    mock_df = pd.DataFrame({"total": [16.2]})

    # check that writing df to csv works, with conversion to csv if incorrect suffix is given
    write_df_to_csv(mock_df, non_csv_path)
    assert not non_csv_path.exists()
    assert csv_path.is_file()


def test_write_dict_to_json(tmp_path: Path) -> None:
    """Happy path for _write_dict_to_json, ensuring writing a dict to a json works.

    Args:
        tmp_path (Path): Path to temporary directory.
    """
    non_json_path = Path(tmp_path, "invalid.txt")
    json_path = Path(tmp_path, "invalid.json")
    mock_dict = {"total": [16.2]}

    # check that writing dict to json works, with conversion if incorrect suffix is given
    write_dict_to_json(mock_dict, non_json_path)
    assert not non_json_path.exists()
    assert json_path.is_file()


@pytest.mark.parametrize("file_name", ["valid.csv", "valid.xlsx"])
def test_load_df_from_file(tmp_path: Path, file_name) -> None:
    """Happy path for "load_df_from_file", ensuring reading a df from a csv / excel works.

    Args:
        tmp_path (Path): Path to temporary directory.
        file_name (str): Name of file.
    """
    file_path = Path(tmp_path, file_name)
    mock_df = pd.DataFrame({"total": [16.2]})

    if file_path.suffix == ".csv":
        mock_df.to_csv(file_path, index=False)
    else:
        pytest.importorskip("openpyxl")
        mock_df.to_excel(file_path, index=False)
    assert file_path.is_file()

    read_df = load_df_from_file(file_path)
    pd.testing.assert_frame_equal(read_df, mock_df)


@pytest.mark.parametrize(
    "file_name, file_type",
    [
        ("valid", ".csv"),  # no suffix - file_type provides it
        ("valid.xlsx", ".csv"),  # wrong suffix - file_type overrides it
    ],
)
def test_load_df_from_file_manual_file_type(
    tmp_path: Path, file_name: str, file_type: Literal[".csv", ".xlsx"]
) -> None:
    """Happy path for "load_df_from_file" when manual file type is provided.

    Args:
        tmp_path (Path): Path to temporary directory.
        file_name (str): Name of file.
        file_type (str): Type of file.
    """
    file_path = Path(tmp_path, file_name)
    mock_df = pd.DataFrame({"total": [16.2]})

    mock_df.to_csv(file_path, index=False)  # always save as CSV content
    assert file_path.is_file()

    read_df = load_df_from_file(file_path, file_type=file_type)
    pd.testing.assert_frame_equal(read_df, mock_df)


def test_load_excel_from_file(tmp_path: Path) -> None:
    """Happy path for "load_excel_from_file", ensuring reading an ExcelFile from an excel works.

    Args:
        tmp_path (Path): Path to temporary directory.
    """
    file_path = Path(tmp_path, "valid.xlsx")
    mock_df = pd.DataFrame({"total": [16.2]})

    pytest.importorskip("openpyxl")
    mock_df.to_excel(file_path, index=False)
    assert file_path.is_file()

    read_excel = load_excel_from_file(file_path)
    read_df = pd.read_excel(read_excel)
    assert isinstance(read_excel, pd.ExcelFile)
    pd.testing.assert_frame_equal(read_df, mock_df)


@pytest.mark.parametrize("helper", [load_df_from_file, load_excel_from_file])
def test_load_from_file_invalid_extension(helper: Callable) -> None:
    """Failure paths for "load_..._from_file" helpers when file has unsupported extensions.

    Args:
        helper (Callable): Helper function to test.
    """
    with pytest.raises(InvalidError, match="Invalid extension"):
        helper("test.txt")


def test_load_df_from_file_invalid_file_type() -> None:
    """Failure path for "load_df_from_file" when an invalid file_type is provided."""
    with pytest.raises(InvalidError, match="Invalid extension"):
        load_df_from_file("test.csv", file_type=".txt")  # type: ignore[arg-type]


@pytest.mark.parametrize("helper", [load_df_from_file, load_excel_from_file])
def test_load_from_file_not_found(helper: Callable) -> None:
    """Failure paths for "load_..._from_file" helpers when file is missing.

    Args:
        helper (Callable): Helper function to test.
    """
    with pytest.raises(InvalidError, match="Invalid path"):
        helper("non_existent_file.xlsx")


@pytest.mark.parametrize(
    "helper, function",
    [(load_df_from_file, "read_excel"), (load_excel_from_file, "ExcelFile")],
)
def test_load_from_file_inaccessible_url(helper: Callable, function: str) -> None:
    """Failure paths for "load_..._from_file" helpers when an inaccessible url is provided.

    Args:
        helper (Callable): Helper function to test.
        function (str): Pandas function used in the helper.
    """
    with patch(
        f"rbc.utils.io.pd.{function}", side_effect=exceptions.HTTPError
    ) as mock_read_excel:
        with pytest.raises(exceptions.HTTPError):
            helper("https://www.website.com/test.xlsx")

    mock_read_excel.assert_called_once()


def test_load_df_from_file_bad_args() -> None:
    """Failure path for "load_df_from_file" when pandas arguments are invalid (TypeError)."""
    with pytest.raises(InvalidError, match="Invalid argument"):
        load_df_from_file("test.csv", sheet_name="Sheet1")


def test_load_excel_from_file_bad_content() -> None:
    """Failure path for "load_excel_from_file" when content is bad (BadZipFile)."""
    with patch(
        "rbc.utils.io.pd.ExcelFile", side_effect=zipfile.BadZipFile
    ) as mock_read_excel:
        with pytest.raises(InvalidError, match="Invalid file content"):
            load_excel_from_file("test.xlsx")

    mock_read_excel.assert_called_once()
