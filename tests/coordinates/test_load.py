# tests/coordinates/test_load.py
"""Tests for reading an operator's raw CSV directories into runs."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from loguru import logger

from rbc.coordinates.load import (
    Run,
    _collect_dirs,
    _get_other_tres_dirs,
    _get_run_name_and_zone,
    _load_ege_rows,
    load_runs,
    recover_names,
)
from rbc.coordinates.mappings import OPERATOR_METADATA
from rbc.coordinates.pipelines.default import DefaultPipeline

ENTSOE_NAME = OPERATOR_METADATA["entsoe"]["name_col"]
ENTSOE_CODE = OPERATOR_METADATA["entsoe"].get("code_col")
ENTSOE_FUEL = OPERATOR_METADATA["entsoe"].get("fuel_col")
ZONE = "10YRO-TEL------P"


# ----------------------------------
# Fixtures
# ----------------------------------
def _write(csv_dir: Path, date: str, rows: list[dict]) -> Path:
    """Write one day's raw rows as a CSV into an operator directory.

    Args:
        csv_dir (Path): The directory to write into (created if missing).
        date (str): The file's date stem, e.g. "2024-01-01".
        rows (list[dict]): The raw rows.

    Returns:
        Path: The written CSV file.
    """
    csv_dir.mkdir(parents=True, exist_ok=True)
    path = Path(csv_dir, f"{date}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def cen_dir(tmp_path: Path) -> Path:
    """A CEN "1h" directory of two days, repeating its EGEs once per timestamp.

    Covers what coordinate finding needs to see: a code-less EGE, an EGE reporting two
    fuel subtypes, and a generation column it doesn't need.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.

    Returns:
        Path: The CEN "1h" directory.
    """
    csv_dir = Path(tmp_path, "cen", "1h")
    for day, sub in (("2024-01-01", "Diésel"), ("2024-01-02", "Gas Natural")):
        _write(
            csv_dir,
            day,
            [
                {
                    "id_central": 1,
                    "central": "TER ALPHA",
                    "tipo_tecnologia": "Térmica",
                    "subtipo_tecnologia": sub,
                    "generation": hour,
                }
                for hour in range(3)
            ]
            + [
                {
                    "id_central": None,
                    "central": "BESS BETA",
                    "tipo_tecnologia": "BESS",
                    "subtipo_tecnologia": None,
                    "generation": hour,
                }
                for hour in range(3)
            ],
        )
    return csv_dir


@pytest.fixture
def entsoe_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """ENTSO-E "15min" and "1h" directories of one zone, an EGE named only in "1h".

    Args:
        tmp_path (Path): Pytest-provided temporary directory.

    Returns:
        tuple[Path, Path]: The "15min" and the "1h" zone directories.
    """
    unnamed = {ENTSOE_NAME: None, ENTSOE_CODE: "30W-UNNAMED", ENTSOE_FUEL: "B14"}
    named = {ENTSOE_NAME: "CET_MINT2", ENTSOE_CODE: "30W-NAMED", ENTSOE_FUEL: "B14"}
    dir_15min = Path(tmp_path, "entsoe", "15min", ZONE)
    dir_1h = Path(tmp_path, "entsoe", "1h", ZONE)
    _write(dir_15min, "2024-01-01", [unnamed, unnamed, named])
    _write(dir_1h, "2024-01-01", [{**unnamed, ENTSOE_NAME: "CET_MINT5"}, named])
    return dir_15min, dir_1h


# ----------------------------------
# Tests: Run and its identity
# ----------------------------------
@pytest.mark.parametrize(
    "parts, expected",
    [
        (("entsoe", "15min", ZONE), (f"15min_{ZONE}", ZONE)),
        (("ons", "1h"), ("1h", None)),
    ],
)
def test_run_name_and_zone(
    tmp_path: Path, parts: tuple[str, ...], expected: tuple[str, str | None]
) -> None:
    """Happy path: a zone directory names its run by resolution and zone, else by resolution.

    The name is also the output files' name, so it has to stay what it was before runs
    existed ("coordinates_15min_10YRO-TEL------P.csv"), or comparisons across runs break.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.
        parts (tuple[str, ...]): The directory's path parts below `tmp_path`.
        expected (tuple[str, str | None]): The expected run name and zone.
    """
    assert _get_run_name_and_zone(Path(tmp_path, *parts)) == expected


def test_map_label_names_the_operator() -> None:
    """Happy path: a run's map label prefixes its name with the operator."""
    run = Run(
        sysop="entsoe", name=f"15min_{ZONE}", zone=ZONE, files=(), df=pd.DataFrame()
    )
    assert run.map_label == f"entsoe_15min_{ZONE}"


# ----------------------------------
# Tests: collecting directories
# ----------------------------------
def test_collect_dirs_filters_correct_files(tmp_path: Path) -> None:
    """Happy path for '_collect_dirs', ensuring correct filtering of files.

    Args:
        tmp_path (Path): Path object to be used as a temporary directory.
    """
    # valid path
    tres_dir = Path(tmp_path, "1h")  # correct tres
    tres_dir.mkdir()
    valid_csv = Path(tres_dir, "2023-01-01.csv")  # correct date
    valid_csv.write_text("data")

    # invalid path
    invalid_csv = Path(tres_dir, "invalid_name.csv")
    invalid_csv.write_text("data")

    with (
        patch("rbc.coordinates.load.DATE_PATTERN") as mock_date,
        patch("rbc.coordinates.load.TRES_PATTERN") as mock_tres,
    ):
        mock_date.match.side_effect = lambda stem: stem == "2023-01-01"
        mock_tres.match.side_effect = lambda part: part == "1h"

        result = _collect_dirs([tmp_path])

        assert result == [tres_dir.resolve()]


def test_collect_dirs_handles_invalid_paths() -> None:
    """Failure path for '_collect_dirs' when the given path does not exist."""
    captured_logs = []
    sink_id = logger.add(lambda msg: captured_logs.append(msg.record), level="WARNING")

    try:
        non_existent_path = Path("/path/does/not/exist")
        result = _collect_dirs([non_existent_path])

        assert result == []
        assert len(captured_logs) == 1
        assert "is not a directory" in captured_logs[0]["message"]
    finally:
        logger.remove(sink_id)


def test_other_tres_dirs_swap_the_resolution(entsoe_dirs: tuple[Path, Path]) -> None:
    """Happy path: a zone's other resolutions are found by swapping the resolution part.

    Args:
        entsoe_dirs (tuple[Path, Path]): The "15min" and the "1h" zone directories.
    """
    dir_15min, dir_1h = entsoe_dirs
    assert _get_other_tres_dirs(dir_15min) == [dir_1h]


# ----------------------------------
# Tests: loading runs
# ----------------------------------
def test_load_runs_builds_one_run_per_directory(cen_dir: Path) -> None:
    """Happy path: a run identifies its directory and lists its CSVs in date order.

    Args:
        cen_dir (Path): The CEN "1h" directory.
    """
    runs = list(load_runs([cen_dir.parent], sysop="cen"))

    assert len(runs) == 1
    assert (runs[0].sysop, runs[0].name, runs[0].zone) == ("cen", "1h", None)
    assert [f.name for f in runs[0].files] == ["2024-01-01.csv", "2024-01-02.csv"]


def test_load_runs_ignores_other_operators(cen_dir: Path) -> None:
    """Happy path: only directories below the operator's own folder become its runs.

    Args:
        cen_dir (Path): The CEN "1h" directory.
    """
    assert list(load_runs([cen_dir.parent.parent], sysop="ons")) == []


def test_load_runs_skips_a_directory_it_cant_load(
    cen_dir: Path, tmp_path: Path
) -> None:
    """Failure path: a directory that can't be read is skipped, the others still run.

    Args:
        cen_dir (Path): The CEN "1h" directory.
        tmp_path (Path): Pytest-provided temporary directory.
    """
    _write(Path(tmp_path, "cen", "15min"), "2024-01-01", [{"central": "TER ALPHA"}])

    def load(files: tuple[Path, ...], sysop: str) -> pd.DataFrame:
        if files[0].parent.name == "15min":
            raise ValueError("unreadable")
        return pd.DataFrame({"central": ["TER ALPHA"]})

    with patch("rbc.coordinates.load._load_ege_rows", side_effect=load):
        runs = list(load_runs([Path(tmp_path, "cen")], sysop="cen"))

    assert [run.name for run in runs] == ["1h"]


def test_ege_rows_hold_only_distinct_metadata_rows(cen_dir: Path) -> None:
    """Happy path: a run holds the distinct EGE rows, not the time series.

    Only the columns the operator's metadata names are read, so memory grows with the
    number of EGEs rather than timestamps (CEN: 2.3M raw rows -> 1272 EGE rows). Every
    variant the pipeline's dedupe needs survives: code-less EGEs, every fuel subtype.

    Args:
        cen_dir (Path): The CEN "1h" directory.
    """
    df = _load_ege_rows(tuple(sorted(cen_dir.glob("*.csv"))), sysop="cen")

    assert "generation" not in df.columns
    assert len(df) == 3  # ALPHA with each of its two subtypes, and BETA
    assert set(df.loc[df["central"] == "TER ALPHA", "subtipo_tecnologia"]) == {
        "Diésel",
        "Gas Natural",
    }
    assert df.loc[df["central"] == "BESS BETA", "id_central"].isna().all()


def test_ege_rows_dedupe_like_the_full_time_series(cen_dir: Path) -> None:
    """Happy path: deduping the distinct EGE rows gives what deduping every row gives.

    Args:
        cen_dir (Path): The CEN "1h" directory.
    """
    files = tuple(sorted(cen_dir.glob("*.csv")))
    full = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ege_rows = _load_ege_rows(files, sysop="cen")
    run = Run(sysop="cen", name="1h", zone=None, files=files, df=ege_rows)

    def dedupe(r: Run) -> pd.DataFrame:
        pipeline = DefaultPipeline(run=r)
        return pipeline._step_dedupe(pd.DataFrame())

    pd.testing.assert_frame_equal(dedupe(run), dedupe(replace(run, df=full)))


# ----------------------------------
# Tests: recovering names from other resolutions
# ----------------------------------
def test_recover_names_from_other_resolution(entsoe_dirs: tuple[Path, Path]) -> None:
    """Happy path: an EGE unnamed in one resolution gets its name from another.

    ENTSO-E's RO 15min data omits names its 1h data has (e.g. "CET_MINT2_CA"), and an
    unnamed EGE can't be matched by name.

    Args:
        entsoe_dirs (tuple[Path, Path]): The "15min" and the "1h" zone directories.
    """
    dir_15min, _ = entsoe_dirs
    df = pd.read_csv(Path(dir_15min, "2024-01-01.csv"))

    out = recover_names(df, dir_15min, "entsoe", run=f"15min_{ZONE}")

    unnamed = out[out[ENTSOE_CODE] == "30W-UNNAMED"]
    assert (unnamed[ENTSOE_NAME] == "CET_MINT5").all()


def test_recover_names_leaves_named_eges_alone(entsoe_dirs: tuple[Path, Path]) -> None:
    """Happy path: an EGE already named in its own rows keeps that name.

    Args:
        entsoe_dirs (tuple[Path, Path]): The "15min" and the "1h" zone directories.
    """
    dir_15min, _ = entsoe_dirs
    df = pd.read_csv(Path(dir_15min, "2024-01-01.csv"))

    out = recover_names(df, dir_15min, "entsoe")

    named = out[out[ENTSOE_CODE] == "30W-NAMED"]
    assert (named[ENTSOE_NAME] == "CET_MINT2").all()


def test_recover_names_without_other_resolutions(tmp_path: Path) -> None:
    """Failure path: with no other resolution to look in, the rows stay as they are.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.
    """
    only_dir = Path(tmp_path, "entsoe", "15min", ZONE)
    unnamed = {ENTSOE_NAME: None, ENTSOE_CODE: "30W-UNNAMED", ENTSOE_FUEL: "B14"}
    _write(only_dir, "2024-01-01", [unnamed])
    df = pd.read_csv(Path(only_dir, "2024-01-01.csv"))

    out = recover_names(df, only_dir, "entsoe")

    assert out[ENTSOE_NAME].isna().all()
