"""ICECHUNK STORE.

Write an operator's generation time series into its versioned Icechunk repository (Zarr
v3), in MW, on a regular UTC axis at the data's own temporal resolution (e.g. hourly,
15-minutely), one repository per resolution and one tagged snapshot per release.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import icechunk as ic
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

# A raw data directory's temporal resolution, as DownloadTask names it (e.g. "15min")
TRES_PATTERN = re.compile(r"(\d+)(min|h|d)")
# Map: the resolution's unit → its ISO 8601 duration, e.g. "15min" → "PT15M"
ISO_DURATIONS = {"min": "PT{}M", "h": "PT{}H", "d": "P{}D"}


@dataclass(frozen=True)
class StorageInfo:
    """Where a repository lives, as the STAC Items' data asset names it.

    Readers never reach the storage directly: a published repository is named by its
    permanent data URL, which energybench's reception redirects to the storage's current
    temporary links.

    Attributes:
        prefix (str): Repository prefix in the bucket, e.g. "energy/cen/PT1H/repository".
        bucket (str | None): S3 bucket, None for a local repository.
        data_base_url (str | None): Permanent public data URL, e.g.
            "https://data.energybench.org"; None for a local repository.
        local_dir (Path | None): Directory of a local repository, None on S3.
    """

    prefix: str
    bucket: str | None = None
    data_base_url: str | None = None
    local_dir: Path | None = None

    @property
    def href(self) -> str:
        """Give the repository's permanent URL (or its path, if local).

        Returns:
            str: e.g. "https://data.energybench.org/energybench/energy/cen/PT1H/repository",
                opened with `icechunk.http_storage(href)`.
        """
        if self.local_dir is not None:
            return str(Path(self.local_dir, self.prefix))
        return f"{str(self.data_base_url).rstrip('/')}/{self.bucket}/{self.prefix}"


def resolution(tres: str) -> tuple[pd.Timedelta, str]:
    """Read a raw data directory's temporal resolution.

    Args:
        tres (str): The directory's name, as DownloadTask names it, e.g. "1h", "15min".

    Returns:
        tuple[pd.Timedelta, str]: The time step, and its ISO 8601 duration (e.g.
            "PT1H", "PT15M"), as repository prefixes and STAC name it.

    Raises:
        ValueError: If the name isn't a temporal resolution.
    """
    if not (match := TRES_PATTERN.fullmatch(tres)):
        raise ValueError(
            f"'{tres}' isn't a temporal resolution such as '1h' or '15min'."
        )
    count, unit = match.groups()
    return pd.Timedelta(tres), ISO_DURATIONS[unit].format(int(count))


def repo_prefix(source: str, tres: str) -> str:
    """Give an operator's repository prefix for one temporal resolution.

    Args:
        source (str): Name of the system operator, e.g. "entsoe".
        tres (str): Name of its raw data directory, e.g. "15min".

    Returns:
        str: e.g. "energy/entsoe/PT15M/repository".
    """
    return f"energy/{source}/{resolution(tres)[1]}/repository"


def open_repository(
    prefix: str,
    local_dir: Path | None = None,
    endpoint: str | None = None,
    bucket: str | None = None,
    s3_region: str = "us-east-1",
) -> ic.Repository:
    """Open (or create) a repository on S3, or in a local directory.

    Args:
        prefix (str): Repository prefix, e.g. "energy/cen/PT1H/repository".
        local_dir (Path | None, optional): Directory for a local repository. Defaults to
            None, which stores it on S3.
        endpoint (str | None, optional): S3 endpoint to write to. Defaults to None.
        bucket (str | None, optional): S3 bucket. Defaults to None.
        s3_region (str, optional): Region the S3 requests are signed for, as the storage
            expects it. Defaults to "us-east-1".

    Returns:
        ic.Repository: The opened repository.

    Raises:
        ValueError: If neither a local directory nor an S3 endpoint and bucket is given,
            or if the S3 credentials are not set.
    """
    if local_dir is not None:
        path = Path(local_dir, prefix)
        path.mkdir(parents=True, exist_ok=True)
        return ic.Repository.open_or_create(ic.local_filesystem_storage(str(path)))

    if not (endpoint and bucket):
        raise ValueError("An S3 repository needs an endpoint and a bucket.")
    key, secret = os.environ.get("S3_ACCESS_KEY"), os.environ.get("S3_SECRET_KEY")
    if not (key and secret):
        raise ValueError("Set S3_ACCESS_KEY and S3_SECRET_KEY to write to S3.")

    storage = ic.s3_storage(
        bucket=bucket,
        prefix=prefix,
        endpoint_url=endpoint,
        region=s3_region,
        allow_http=endpoint.startswith("http://"),
        force_path_style=True,
        access_key_id=key,
        secret_access_key=secret,
    )
    return ic.Repository.open_or_create(storage)


class GenerationStore:
    """Writes one release of an operator's generation into its Icechunk repository.

    The values arrive in time order (one raw file after the other). They are buffered
    one time chunk at a time, and every complete chunk is written as one region, so
    memory holds a single chunk row of all EGEs. Time steps without a value stay NaN.

    If the release's tag already exists, nothing is written: the values are still taken
    in (for each EGE's time span), but the tag's snapshot is reused.

    Attributes:
        repo (ic.Repository): The repository written to.
        tag (str): The release's tag, e.g. "v0.1.0".
        reused (bool): Whether the tag already existed, so nothing is written.
        entities (list[str]): The EGE IDs along `entity_id`, in order.
        step (pd.Timedelta): The time step, e.g. 1 hour or 15 minutes.
        times (pd.DatetimeIndex): The regular UTC axis (tz-naive), in order.
        snapshot (str | None): The tag's snapshot ID, once written (or reused).
    """

    def __init__(
        self,
        repo: ic.Repository,
        tag: str,
        entities: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        step: pd.Timedelta,
        chunks: tuple[int, int],
    ):
        """Create the release's empty array, or find its existing tag.

        Args:
            repo (ic.Repository): The repository to write to.
            tag (str): The release's tag, e.g. "v0.1.0".
            entities (list[str]): All EGE IDs the release holds.
            start (pd.Timestamp): The first time step's start (UTC).
            end (pd.Timestamp): The last time step's end (UTC), exclusive.
            step (pd.Timedelta): The time step, e.g. 1 hour or 15 minutes.
            chunks (tuple[int, int]): Chunk shape (EGEs, time steps).

        Raises:
            ValueError: If the EGE IDs repeat, or an existing tag holds other EGEs.
        """
        if len(set(entities)) != len(entities):
            raise ValueError("EGE IDs along 'entity_id' must be unique.")

        self.repo = repo
        self.tag = tag
        self.entities = list(entities)
        self.step = step
        self.times = pd.date_range(
            _utc_naive(start), _utc_naive(end), freq=step, inclusive="left"
        )
        self._chunk_steps = chunks[1]
        self._entity_index = pd.Index(self.entities)

        # each EGE's first and last time step with a value (-1 = none yet)
        self._first = np.full(len(self.entities), -1, dtype=np.int64)
        self._last = np.full(len(self.entities), -1, dtype=np.int64)

        self._chunk = 0  # index of the time chunk being buffered
        self._buffer = self._empty_buffer()

        self.snapshot: str | None = None
        self._session: ic.Session | None = None
        self.reused = tag in set(repo.list_tags())  # nothing new is written
        if self.reused:
            self.snapshot = repo.lookup_tag(tag)
            self._check_existing_entities()
            logger.warning(
                f"Release '{tag}' already exists (snapshot {self.snapshot}): reusing "
                f"it, no values are written. Raise the release to write new data."
            )
        else:
            self._session = repo.writable_session("main")
            template = xr.Dataset(
                {
                    "values": (
                        ("entity_id", "time"),
                        da.full(
                            (len(self.entities), len(self.times)),
                            np.nan,
                            dtype="float32",
                            chunks=chunks,
                        ),
                        {"units": "MW"},
                    )
                },
                coords={
                    "entity_id": np.array(self.entities, dtype=object),
                    "time": self.times,
                },
            )
            # compute=False: writes the coordinates and the empty array's metadata only
            template.to_zarr(
                self._session.store,
                zarr_format=3,
                consolidated=False,
                mode="w",
                compute=False,
                encoding={"values": {"chunks": chunks}},
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(self, ids: pd.Series, timestamps: pd.Series, values: pd.Series) -> None:
        """Take in values, each for one EGE and time step, in time order across calls.

        Args:
            ids (pd.Series): The EGEs' IDs.
            timestamps (pd.Series): The time steps' starts (UTC, tz-aware).
            values (pd.Series): The values in MW (NaN where unknown).

        Raises:
            ValueError: If an ID isn't one of the release's EGEs, a timestamp isn't on
                the axis or lies before the chunk being buffered, or an EGE's time step
                already has a value (the same step reported twice).
        """
        if ids.empty:
            return

        ent = self._entity_index.get_indexer(ids)
        if (unknown := ent < 0).any():
            raise ValueError(
                f"EGE IDs not in the release: {sorted(set(ids[unknown]))[:5]}"
            )
        steps = self.times.get_indexer(_utc_naive(timestamps))
        if (outside := steps < 0).any():
            raise ValueError(
                f"Timestamps not on the release's time axis: "
                f"{sorted(set(timestamps[outside].astype(str)))[:5]}"
            )

        vals = values.to_numpy(dtype="float32")
        chunk_of = steps // self._chunk_steps
        if chunk_of.min() < self._chunk:
            raise ValueError("Values arrived out of time order (an earlier chunk).")

        for chunk in np.unique(chunk_of):
            if chunk > self._chunk:
                self._flush()
                self._chunk = int(chunk)
            sel = chunk_of == chunk
            self._put(ent[sel], steps[sel], vals[sel])

    def finish(self, message: str) -> str:
        """Write the last chunk, commit and tag the release (unless it already existed).

        Args:
            message (str): The commit message.

        Returns:
            str: The snapshot ID the release's tag points to.
        """
        if self._session is None:  # reused existing tag
            assert self.snapshot is not None
            return self.snapshot

        self._flush()
        self.snapshot = self._session.commit(message)
        self.repo.create_tag(self.tag, self.snapshot)
        self._session = None
        logger.info(f"Committed release '{self.tag}' as snapshot {self.snapshot}.")
        return self.snapshot

    def spans(self) -> pd.DataFrame:
        """Give each EGE's first and last time step with a value, as far as taken in.

        Returns:
            pd.DataFrame: Indexed by EGE ID, with the columns "start" and "end" (UTC,
                tz-aware; the end is the last step's end), missing where no value.
        """
        has = self._first >= 0
        start = pd.Series(pd.NaT, index=self.entities, dtype="datetime64[ns, UTC]")
        end = start.copy()
        start[has] = self.times[self._first[has]].tz_localize("UTC")
        end[has] = (self.times[self._last[has]] + self.step).tz_localize("UTC")
        return pd.DataFrame({"start": start, "end": end})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _put(self, ent: np.ndarray, steps: np.ndarray, vals: np.ndarray) -> None:
        """Place values into the buffer of the current chunk, refusing double steps.

        Args:
            ent (np.ndarray): The values' EGE indices.
            steps (np.ndarray): The values' time step indices (within the current chunk).
            vals (np.ndarray): The values.

        Raises:
            ValueError: If an EGE's time step is given twice (in this call or before).
        """
        cols = steps - self._chunk * self._chunk_steps
        pairs = ent.astype(np.int64) * self._chunk_steps + cols
        if (
            len(np.unique(pairs)) < len(pairs)
            or (~np.isnan(self._buffer[ent, cols])).any()
        ):
            raise ValueError(
                "An EGE's time step was given twice (e.g. a DST hour converted "
                "wrongly)."
            )
        self._buffer[ent, cols] = vals

        # each EGE's first and last valued step among these, merged into the spans so far
        valued = ~np.isnan(vals)
        lo = np.full(len(self.entities), np.iinfo(np.int64).max, dtype=np.int64)
        hi = np.full(len(self.entities), -1, dtype=np.int64)
        np.minimum.at(lo, ent[valued], steps[valued])
        np.maximum.at(hi, ent[valued], steps[valued])
        seen = hi >= 0
        self._first[seen] = np.where(
            self._first[seen] < 0, lo[seen], np.minimum(self._first[seen], lo[seen])
        )
        self._last = np.maximum(self._last, hi)

    def _flush(self) -> None:
        """Write the buffered chunk as one region and start an empty one."""
        start = self._chunk * self._chunk_steps
        stop = min(start + self._chunk_steps, len(self.times))
        if self._session is not None and start < stop:
            region = self._buffer[:, : stop - start]
            if not np.isnan(region).all():
                xr.Dataset({"values": (("entity_id", "time"), region)}).to_zarr(
                    self._session.store,
                    region={
                        "entity_id": slice(0, len(self.entities)),
                        "time": slice(start, stop),
                    },
                    consolidated=False,
                )
        self._buffer = self._empty_buffer()

    def _empty_buffer(self) -> np.ndarray:
        """Give an all-NaN buffer for one time chunk of all EGEs.

        Returns:
            np.ndarray: Shaped (EGEs, chunk time steps), float32.
        """
        return np.full((len(self.entities), self._chunk_steps), np.nan, dtype="float32")

    def _check_existing_entities(self) -> None:
        """Check the existing tag holds exactly this release's EGEs.

        Raises:
            ValueError: If the tag's EGEs differ from this run's.
        """
        session = self.repo.readonly_session(tag=self.tag)
        stored = xr.open_zarr(session.store, consolidated=False)["entity_id"].values
        if [str(e) for e in stored] != self.entities:
            raise ValueError(
                f"Release '{self.tag}' exists with other EGEs: raise the release."
            )


def _utc_naive(timestamps: pd.Series | pd.Timestamp) -> pd.DatetimeIndex | pd.Timestamp:
    """Give UTC timestamps without their time zone, as the Zarr time axis stores them.

    Args:
        timestamps (pd.Series | pd.Timestamp): tz-aware (any zone) or naive UTC times.

    Returns:
        pd.DatetimeIndex | pd.Timestamp: The naive UTC times.
    """
    if isinstance(timestamps, pd.Timestamp):
        if timestamps.tz is None:
            return timestamps
        return timestamps.tz_convert("UTC").tz_localize(None)
    index = pd.DatetimeIndex(timestamps)
    return index.tz_convert("UTC").tz_localize(None) if index.tz else index
