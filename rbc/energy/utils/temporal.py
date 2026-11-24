"""TEMPORAL.

Time handling shared by the energy processors (local/UTC timestamps, periods, resolutions).
"""

from typing import Literal

import pandas as pd

from rbc.energy.utils.downloading import DATE_PATTERN, TRES_PATTERN

# Map: the resolution's unit → its ISO 8601 duration, e.g. "15min" → "PT15M"
ISO_DURATIONS = {"min": "PT{}M", "h": "PT{}H", "d": "P{}D"}


def to_utc(
    timestamps: pd.Series,
    tz: str,
    dst: pd.Series,
    skipped: Literal["raise", "NaT"] = "raise",
) -> pd.Series:
    """Convert naive local timestamps to tz-aware UTC, keeping every hour.

    The time zone applies its country's DST rules, past ones included. The hour a DST
    change repeats (clocks set back) is told apart by `dst`. An hour a DST change skips
    (clocks set forward) can't hold real data, so it raises instead of being guessed,
    unless the caller handles such rows itself (`skipped="NaT"`).

    Args:
        timestamps (pd.Series): Naive local timestamps, e.g. '2024-01-01 00:00:00'.
        tz (str): The operator's IANA time zone, e.g. "America/Santiago".
        dst (pd.Series): Whether each timestamp is daylight saving time. Only read where
            the local time is repeated, e.g. True for the first of its two rows.
        skipped (Literal["raise", "NaT"], optional): What a timestamp in an hour a DST
            change skips gets: an error, or NaT for the caller to handle. Defaults to
            "raise".

    Returns:
        pd.Series: The timestamps in UTC (tz-aware), none missing (unless skipped).

    Raises:
        ValueError: If a timestamp is missing or (by default) lies in an hour a DST
            change skips.
    """
    local = pd.to_datetime(timestamps)
    if missing := int(local.isna().sum()):
        raise ValueError(f"{missing} timestamps are missing.")

    aware = local.dt.tz_localize(
        tz, ambiguous=dst.to_numpy(dtype=bool), nonexistent="NaT"
    )
    if skipped == "raise" and (in_gap := aware.isna()).any():
        raise ValueError(
            f"{int(in_gap.sum())} timestamps lie in an hour a DST change skips ({tz}), "
            f"e.g. {sorted(local[in_gap].astype(str).unique())[:5]}."
        )
    return aware.dt.tz_convert("UTC")


def get_period_bounds(period: str, tz: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Get a local year, month or day's start and end in UTC (e.g. a raw file's period).

    Args:
        period (str): The local period, e.g. '2025'/'2018-11'/'2023-04-01' (s. DATE_PATTERN).
        tz (str): The operator's IANA time zone, e.g. "America/Santiago".

    Returns:
        tuple[pd.Timestamp, pd.Timestamp]: The period's start and end (the next period's
            start) in UTC. If a DST change skips local midnight (clocks jump from 00:00
            to 01:00), that boundary is 01:00, the first local time that exists.

    Raises:
        ValueError: If the period isn't a year, month or day.
    """
    if not DATE_PATTERN.match(period):
        raise ValueError(f"'{period}' isn't a year, month or day (e.g. '2025-07').")

    first = pd.Timestamp(period)
    following = {
        4: pd.offsets.YearBegin(1),
        7: pd.offsets.MonthBegin(1),
        10: pd.Timedelta(days=1),
    }[len(period)]
    start, end = (
        d.tz_localize(tz, nonexistent="shift_forward").tz_convert("UTC")
        for d in (first, first + following)
    )
    return start, end


def get_resolution(tres: str) -> tuple[pd.Timedelta, str]:
    """Read a raw data directory's temporal resolution.

    Args:
        tres (str): The directory's name, as DownloadTask names it, e.g. "1h", "15min".

    Returns:
        tuple[pd.Timedelta, str]: The time step (`step`) to compute with, and the
            resolution as an ISO 8601 duration (`tres_iso`, e.g. "PT1H", "PT15M"), as
            repository prefixes and STAC name it.

    Raises:
        ValueError: If the name isn't a temporal resolution.
    """
    if not (match := TRES_PATTERN.fullmatch(tres)):
        raise ValueError(
            f"'{tres}' isn't a temporal resolution such as '1h' or '15min'."
        )
    count, unit = match.groups()
    step = pd.Timedelta(int(count), unit="D" if unit == "d" else unit)  # 'd' deprecated
    return step, ISO_DURATIONS[unit].format(int(count))
