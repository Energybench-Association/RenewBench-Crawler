# tests/publish/test_icechunk_store.py
"""Tests for writing a release's generation into its Icechunk repository."""

from pathlib import Path
from unittest.mock import MagicMock

import icechunk as ic
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rbc.publish.icechunk_store import (
    EnergyIcechunkWriter,
    StorageInfo,
    get_resolution,
    get_storage_location,
    open_or_create_repository,
)

START = pd.Timestamp("2024-01-01 00:00", tz="UTC")
END = pd.Timestamp("2024-01-01 10:00", tz="UTC")


def _hours(*hours: int) -> pd.Series:
    """UTC hours after START.

    Args:
        *hours (int): Hours after START.

    Returns:
        pd.Series: The tz-aware timestamps.
    """
    return pd.Series([START + pd.Timedelta(hours=h) for h in hours])


def _writer(
    tmp_path: Path, tag: str = "v0.1.0", entities=("a", "b"), end: pd.Timestamp = END
) -> EnergyIcechunkWriter:
    """A writer of two EGEs over 10 hours, in chunks of 4 hours.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.
        tag (str, optional): The release. Defaults to "v0.1.0".
        entities (tuple, optional): The EGE IDs. Defaults to ("a", "b").
        end (pd.Timestamp, optional): The time axis' end. Defaults to END.

    Returns:
        EnergyIcechunkWriter: The writer.
    """
    repo = open_or_create_repository(
        StorageInfo("energy/x/PT1H/repository", local_dir=tmp_path)
    )
    return EnergyIcechunkWriter(
        repo, tag, list(entities), START, end, step=pd.Timedelta("1h"), chunks=(1, 4)
    )


def _read(tmp_path: Path, tag: str = "v0.1.0") -> xr.Dataset:
    """Read a release back.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.
        tag (str, optional): The release. Defaults to "v0.1.0".

    Returns:
        xr.Dataset: The release's data.
    """
    path = Path(tmp_path, "energy/x/PT1H/repository")
    repo = ic.Repository.open(ic.local_filesystem_storage(str(path)))
    return xr.open_zarr(repo.readonly_session(tag=tag).store, consolidated=False).load()


def test_values_land_on_their_ege_and_hour(tmp_path: Path) -> None:
    """Happy path: values across chunks are written, and unreported hours stay NaN."""
    writer = _writer(tmp_path)
    writer.add(pd.Series(["a", "b"]), _hours(0, 1), pd.Series([1.0, 2.0]))
    writer.add(pd.Series(["a"]), _hours(9), pd.Series([3.0]))  # last, partial chunk
    snapshot = writer.finish("test")

    ds = _read(tmp_path)
    assert ds["values"].sel(entity_id="a").values[[0, 9]].tolist() == [1.0, 3.0]
    assert ds["values"].sel(entity_id="b").values[1] == 2.0
    assert int(ds["values"].notnull().sum()) == 3
    assert ds.sizes == {"entity_id": 2, "time": 10}
    assert snapshot == ic.Repository.open(
        ic.local_filesystem_storage(str(Path(tmp_path, "energy/x/PT1H/repository")))
    ).lookup_tag("v0.1.0")


def test_spans_are_each_eges_first_and_last_hour(tmp_path: Path) -> None:
    """Happy path: an EGE's span ends with its last hour; without values it has none."""
    writer = _writer(tmp_path, entities=("a", "b", "c"))
    writer.add(
        pd.Series(["a", "a", "b"]), _hours(2, 6, 3), pd.Series([1.0, 1.0, np.nan])
    )

    spans = writer.spans()
    assert spans.loc["a"].tolist() == [
        START + pd.Timedelta(hours=2),
        START + pd.Timedelta(hours=7),
    ]
    assert spans.loc[["b", "c"]].isna().all().all()  # a NaN value isn't a value


@pytest.mark.parametrize(
    "ids, hours, match",
    [
        (["z"], [0], "not in the release"),
        (["a"], [10], "not on the release's time axis"),
        (["a", "a"], [1, 1], "given twice"),
    ],
)
def test_bad_values_raise(tmp_path: Path, ids: list, hours: list, match: str) -> None:
    """Failure path: an unknown EGE, an hour off the axis or a double hour raises.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.
        ids (list): The values' EGE IDs.
        hours (list): Their hours after START.
        match (str): The expected error.
    """
    with pytest.raises(ValueError, match=match):
        _writer(tmp_path).add(
            pd.Series(ids), _hours(*hours), pd.Series([1.0] * len(ids))
        )


def test_double_hour_across_calls_and_earlier_chunk_raise(tmp_path: Path) -> None:
    """Failure path: an hour given again later, or values going back a chunk, raise."""
    writer = _writer(tmp_path)
    writer.add(pd.Series(["a"]), _hours(1), pd.Series([1.0]))
    with pytest.raises(ValueError, match="given twice"):
        writer.add(pd.Series(["a"]), _hours(1), pd.Series([1.0]))
    writer.add(pd.Series(["a"]), _hours(5), pd.Series([1.0]))
    with pytest.raises(ValueError, match="out of time order"):
        writer.add(pd.Series(["b"]), _hours(0), pd.Series([1.0]))


def test_existing_release_is_reused(tmp_path: Path) -> None:
    """Happy path: a release whose tag exists keeps its snapshot, nothing is rewritten."""
    first = _writer(tmp_path)
    first.add(pd.Series(["a"]), _hours(0), pd.Series([1.0]))
    snapshot = first.finish("first")

    again = _writer(tmp_path)
    assert again.reused and not first.reused
    again.add(pd.Series(["a"]), _hours(0), pd.Series([99.0]))
    assert again.finish("again") == snapshot
    assert _read(tmp_path)["values"].sel(entity_id="a").values[0] == 1.0
    assert again.spans().loc["a", "start"] == START


def test_existing_release_with_other_eges_raises(tmp_path: Path) -> None:
    """Failure path: reusing a release that holds other EGEs raises."""
    _writer(tmp_path).finish("first")
    with pytest.raises(ValueError, match="other EGEs"):
        _writer(tmp_path, entities=("a", "c"))


def test_existing_release_with_another_time_axis_raises(tmp_path: Path) -> None:
    """Failure path: reusing a release over another time span raises.

    E.g. a test run of one year, then a full run under the same release: its records
    would claim the years the tag doesn't hold.
    """
    _writer(tmp_path).finish("first")
    with pytest.raises(ValueError, match="another time axis"):
        _writer(tmp_path, end=END + pd.Timedelta(hours=5))


def test_storage_location_and_href(tmp_path: Path) -> None:
    """Happy path: a release's repository is published or local, and where readers find it."""
    prefix = "energy/cen/PT1H/repository"
    cfg = MagicMock(bucket="energybench", url_public="https://data.example.org/")

    published = get_storage_location(prefix, cfg, publish=True, local_dir=tmp_path)
    assert published.local_dir is None
    assert (
        published.href
        == "https://data.example.org/energybench/energy/cen/PT1H/repository"
    )
    local = get_storage_location(prefix, cfg, publish=False, local_dir=tmp_path)
    assert local.href == str(Path(tmp_path, prefix))


def test_published_repository_on_the_lsdf_route(tmp_path: Path) -> None:
    """Happy path: on the LSDF route, a published repository goes into dst_dir_data."""
    cfg = MagicMock(route="lsdf", bucket="b")
    cfg.lsdf.dst_dir_data = tmp_path
    open_or_create_repository(StorageInfo("p", bucket="b"), cfg)
    assert Path(tmp_path, "b", "p").is_dir()


def test_published_repository_needs_config_and_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure path: publishing without the config or S3 credentials raises."""
    storage = StorageInfo("p", bucket="b")
    with pytest.raises(ValueError, match="needs the publish config"):
        open_or_create_repository(storage)

    monkeypatch.delenv("S3_ACCESS_KEY", raising=False)
    cfg = MagicMock(route="s3", bucket="b")
    cfg.s3.url_endpoint = "https://s3.example.org"
    with pytest.raises(ValueError, match="S3_ACCESS_KEY"):
        open_or_create_repository(storage, cfg)


@pytest.mark.parametrize(
    "tres, step, iso",
    [
        ("1h", pd.Timedelta(hours=1), "PT1H"),
        ("15min", pd.Timedelta(minutes=15), "PT15M"),
        ("5min", pd.Timedelta(minutes=5), "PT5M"),
        ("30min", pd.Timedelta(minutes=30), "PT30M"),
        ("1d", pd.Timedelta(days=1), "P1D"),
    ],
)
def test_get_resolution(tres: str, step: pd.Timedelta, iso: str) -> None:
    """Happy path: a raw data dir's name gives the time step and its ISO 8601 duration.

    Args:
        tres (str): The dir's name, as DownloadTask names it.
        step (pd.Timedelta): The expected time step.
        iso (str): The expected ISO 8601 duration.
    """
    assert get_resolution(tres) == (step, iso)


def test_get_resolution_raises_on_other_names() -> None:
    """Failure path: a directory name that isn't a temporal resolution raises."""
    with pytest.raises(ValueError, match="isn't a temporal resolution"):
        get_resolution("hourly")


def test_15_minute_axis(tmp_path: Path) -> None:
    """Happy path: a 15-minute writer keeps every quarter hour; off-axis times raise."""
    repo = open_or_create_repository(
        StorageInfo("energy/x/PT15M/repository", local_dir=tmp_path)
    )
    writer = EnergyIcechunkWriter(
        repo,
        "v0.1.0",
        ["a"],
        START,
        START + pd.Timedelta(hours=2),
        step=pd.Timedelta("15min"),
        chunks=(1, 4),
    )
    quarter = [START + pd.Timedelta(minutes=15 * i) for i in (0, 1, 5)]
    writer.add(pd.Series(["a"] * 3), pd.Series(quarter), pd.Series([1.0, 2.0, 3.0]))
    writer.finish("15 min")

    path = Path(tmp_path, "energy/x/PT15M/repository")
    ro = ic.Repository.open(ic.local_filesystem_storage(str(path)))
    ds = xr.open_zarr(ro.readonly_session(tag="v0.1.0").store, consolidated=False)
    assert ds.sizes["time"] == 8
    assert ds["values"].sel(entity_id="a").values[[0, 1, 5]].tolist() == [1.0, 2.0, 3.0]
    assert writer.spans().loc["a", "end"] == START + pd.Timedelta(minutes=90)

    with pytest.raises(ValueError, match="not on the release's"):
        writer.add(
            pd.Series(["a"]),
            pd.Series([START + pd.Timedelta(minutes=7)]),
            pd.Series([1.0]),
        )
