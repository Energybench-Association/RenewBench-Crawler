# tests/energy/utils/test_temporal.py
"""Tests for the energy processors' time handling (UTC, raw files' periods, resolutions)."""

import pandas as pd
import pytest

from rbc.energy.utils.temporal import get_period_bounds, get_resolution, to_utc
from rbc.utils.errors import InvalidError


@pytest.mark.parametrize(
    "local, dst, utc",
    [
        ("2024-01-01 00:00:00", True, "2024-01-01 03:00:00"),  # summer: DST, UTC-3
        ("2025-08-01 00:00:00", True, "2025-08-01 04:00:00"),  # winter: UTC-4
        ("2023-04-01 23:00:00", True, "2023-04-02 02:00:00"),  # repeated: 1st, DST
        ("2023-04-01 23:00:00", False, "2023-04-02 03:00:00"),  # repeated: 2nd, std
        ("2015-06-01 00:00:00", True, "2015-06-01 03:00:00"),  # 2015: DST all year
    ],
)
def test_to_utc(local: str, dst: bool, utc: str) -> None:
    """Happy path: local time becomes UTC by the date's own offset, every hour kept.

    The hour clocks are set back is told apart by its DST flag, not left empty.

    Args:
        local (str): A local timestamp in Chile.
        dst (bool): Whether the row is the DST one of a repeated hour.
        utc (str): The expected UTC timestamp.
    """
    out = to_utc(pd.Series([local]), tz="America/Santiago", dst=pd.Series([dst]))
    assert out.tolist() == [pd.Timestamp(utc, tz="UTC")]


def test_to_utc_keeps_both_rows_of_the_repeated_hour() -> None:
    """Happy path: the day clocks are set back has 25 distinct UTC hours."""
    hours = [f"2023-04-01 {h:02d}:00:00" for h in range(24)] + ["2023-04-01 23:00:00"]
    dst = pd.Series(hours).duplicated().map({False: True, True: False})

    out = to_utc(pd.Series(hours), tz="America/Santiago", dst=dst)

    assert out.nunique() == 25
    assert out.min() == pd.Timestamp("2023-04-01 03:00", tz="UTC")  # 00:00, UTC-3
    assert out.max() == pd.Timestamp("2023-04-02 03:00", tz="UTC")  # 2nd 23:00, UTC-4


@pytest.mark.parametrize(
    "local", ["2023-09-03 00:00:00", None], ids=["skipped_hour", "missing"]
)
def test_to_utc_raises_instead_of_dropping(local: str | None) -> None:
    """Failure path: an hour clocks skip, or no timestamp, raises instead of vanishing.

    Args:
        local (str | None): A timestamp no UTC hour exists for.
    """
    with pytest.raises(ValueError):
        to_utc(pd.Series([local]), tz="America/Santiago", dst=pd.Series([True]))


@pytest.mark.parametrize(
    "period, start, end",
    [
        ("2025", "2025-01-01 03:00", "2026-01-01 03:00"),
        ("2018-11", "2018-11-01 03:00", "2018-12-01 02:00"),  # DST from 2018-11-04
        ("2018-11-04", "2018-11-04 03:00", "2018-11-05 02:00"),  # 00:00 skipped: 23 h
    ],
)
def test_get_period_bounds(period: str, start: str, end: str) -> None:
    """Happy path: a year, month or day's local start and end, in UTC (here Brasília).

    Args:
        period (str): The local period.
        start (str): Its expected start in UTC.
        end (str): Its expected end in UTC.
    """
    assert get_period_bounds(period, "America/Sao_Paulo") == (
        pd.Timestamp(start, tz="UTC"),
        pd.Timestamp(end, tz="UTC"),
    )


def test_get_period_bounds_rejects_other_formats() -> None:
    """Failure path: a period that isn't a year, month or day raises."""
    with pytest.raises(InvalidError, match="year, month or day"):
        get_period_bounds("2025-W07", "America/Sao_Paulo")


@pytest.mark.parametrize(
    "tres, step, tres_iso",
    [
        ("1h", pd.Timedelta(hours=1), "PT1H"),
        ("15min", pd.Timedelta(minutes=15), "PT15M"),
        ("5min", pd.Timedelta(minutes=5), "PT5M"),
        ("30min", pd.Timedelta(minutes=30), "PT30M"),
        ("1d", pd.Timedelta(days=1), "P1D"),
    ],
)
def test_get_resolution(tres: str, step: pd.Timedelta, tres_iso: str) -> None:
    """Happy path: a raw data dir's name gives the time step and its ISO 8601 duration.

    Args:
        tres (str): The dir's name, as DownloadTask names it.
        step (pd.Timedelta): The expected time step.
        tres_iso (str): The expected ISO 8601 duration.
    """
    assert get_resolution(tres) == (step, tres_iso)


def test_get_resolution_raises_on_other_names() -> None:
    """Failure path: a directory name that isn't a temporal resolution raises."""
    with pytest.raises(InvalidError, match="isn't a temporal resolution"):
        get_resolution("hourly")
