# tests/energy/utils/test_processing.py
"""Tests for the energy processors' shared base and output layout."""

from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from loguru import logger

from rbc.energy.utils import processing
from rbc.energy.utils.processing import (
    STAC_COLUMNS,
    ZARR_COLUMNS,
    EnergyProcessor,
    check_totals_kept,
    check_zarr_rows,
    log_unjoined_ids,
    to_layout,
)
from rbc.utils.errors import DataStructureError


def test_to_layout_keeps_only_the_layouts_columns() -> None:
    """Happy path: renamed columns are kept in the layout's order, all others dropped."""
    df = pd.DataFrame({"gen_real_mw": [1.0], "fecha_hora": ["2024-01-01"], "hora": [1]})

    out = to_layout(
        df,
        layout=ZARR_COLUMNS,
        renames={"fecha_hora": "timestamp", "gen_real_mw": "value"},
    )

    assert list(out.columns) == ZARR_COLUMNS
    assert out.loc[0, ["timestamp", "value"]].tolist() == ["2024-01-01", 1.0]


def test_to_layout_leaves_what_isnt_provided_unknown() -> None:
    """Happy path: a layout column the operator doesn't provide exists, but stays empty.

    So every operator's output has the same columns, whatever it reports.
    """
    out = to_layout(pd.DataFrame({"central": ["HE ANTUCO"]}), STAC_COLUMNS)

    assert list(out.columns) == STAC_COLUMNS
    assert out.loc[0].isna().all()


def _ids(sysop: str, codes: list, names: list | None = None) -> pd.Series:
    """Give EGEs their global IDs, as a processor of `sysop` does.

    Args:
        sysop (str): The operator, e.g. "cen".
        codes (list): The EGEs' codes (None for missing).
        names (list | None, optional): Their names. Defaults to none.

    Returns:
        pd.Series: The EGEs' global IDs.
    """

    class _OfSysop(_Processor):
        DATASET = MagicMock(sysop=sysop)

    names = names or [None] * len(codes)
    return _OfSysop._get_global_ids(pd.Series(codes), pd.Series(names, dtype="string"))


def test_global_ids_are_the_same_in_every_run() -> None:
    """Happy path: an EGE's global ID depends on its key alone, not on the other EGEs.

    So an EGE added later neither changes the existing IDs nor takes one of them.
    """
    first = _ids("cen", ["116", "BESS BETA"])
    later = _ids("cen", ["2", "BESS BETA", "116"])

    assert first.tolist() == [later[2], later[1]] == ["cen-116", "cen-bess-beta"]
    assert later.nunique() == 3


def test_global_ids_differ_between_operators() -> None:
    """Happy path: the same key of two operators gives two IDs, unique across all."""
    assert _ids("cen", ["1"])[0] != _ids("ons", ["1"])[0]


def test_global_ids_missing_without_key() -> None:
    """Failure path: an EGE without a key gets no ID, rather than a shared one."""
    assert _ids("cen", ["1", None]).isna().tolist() == [False, True]


def test_global_ids_raise_on_clash() -> None:
    """Failure path: two keys written the same way raise, so no two EGEs share an ID."""
    with pytest.raises(ValueError, match="share the ID 'cen-bess-beta'"):
        _ids("cen", ["BESS Beta", "BESS-BETA"])


@pytest.mark.parametrize(
    "code, name, key",
    [
        (116.0, "HE ANTUCO", "116"),
        ("116", "HE ANTUCO [NO_MOSTRAR]", "116"),
        ("AMBA", "ANGRA I", "AMBA"),  # e.g. ONS
        ("48W000000ABC1234", "UNIT 1", "48W000000ABC1234"),  # e.g. ENTSO-E's EIC
        (None, "BESS BETA", "BESS BETA"),
    ],
)
def test_global_ids_by_code_else_name(
    code: float | str | None, name: str, key: str
) -> None:
    """Happy path: an EGE's global ID is keyed by its code as text, else by its name.

    Both tables are written with these IDs, so they have to agree however a numeric code
    was read (116, 116.0, '116'), and a code that isn't a number is a code all the same.

    Args:
        code (float | str | None): The EGE's code, as read.
        name (str): Its name.
        key (str): The key its global ID should derive from.
    """
    assert _ids("cen", [code], [name]).tolist() == _ids("cen", [key]).tolist()


def _zarr_rows(ids: Sequence[str | None], times: Sequence[str | None]) -> pd.DataFrame:
    """A raw file's rows in the Zarr layout.

    Args:
        ids (Sequence[str | None]): The rows' EGE IDs (None for missing).
        times (Sequence[str | None]): Their UTC timestamps (None for missing).

    Returns:
        pd.DataFrame: The rows, with "value" 1.0 each.
    """
    return pd.DataFrame(
        {"id": ids, "timestamp": pd.to_datetime(times, utc=True), "value": 1.0}
    )


START = pd.Timestamp("2024-01-01 03:00", tz="UTC")  # a local midnight, e.g. UTC-3


def test_check_zarr_rows_passes_steps_on_the_grid() -> None:
    """Happy path: each EGE's steps on the grid pass, also outside the file's period.

    CEN reports the hour a DST change repeats in the next day's file, so the period is
    left to the operator's own check.
    """
    times = ["2024-01-01 02:00", "2024-01-01 03:00", "2024-01-01 04:00"]
    check_zarr_rows(
        _zarr_rows(["a", "a", "b"], times), start=START, step=pd.Timedelta(hours=1)
    )


@pytest.mark.parametrize(
    "ids, times, match",
    [
        ([None, "a"], ["2024-01-01 03:00", "2024-01-01 03:00"], "without an EGE ID"),
        (["a", "a"], ["2024-01-01 03:00", "2024-01-01 03:30"], "off the time grid"),
        (["a", "a"], ["2024-01-01 03:00", None], "missing"),
        (["a", "a"], ["2024-01-01 04:00", "2024-01-01 04:00"], "given twice"),
    ],
    ids=["no_id", "off_grid", "missing", "twice"],
)
def test_check_zarr_rows_raises(
    ids: list[str | None], times: list[str | None], match: str
) -> None:
    """Failure path: a missing ID, or a step off the grid, missing or twice raises.

    Args:
        ids (list[str | None]): The rows' EGE IDs.
        times (list[str | None]): Their UTC timestamps.
        match (str): The expected error message part.
    """
    with pytest.raises(DataStructureError, match=match):
        check_zarr_rows(_zarr_rows(ids, times), start=START, step=pd.Timedelta(hours=1))


def test_check_totals_kept_passes_equal_totals() -> None:
    """Happy path: totals equal up to float rounding pass silently."""
    check_totals_kept(
        pd.Series({"a": 0.1 + 0.2, "b": 5.0}), pd.Series({"a": 0.3, "b": 5.0}), "x"
    )


def test_check_totals_kept_raises_on_lost_values() -> None:
    """Failure path: an EGE whose values shrank or vanished raises with its ID."""
    with pytest.raises(ValueError, match="2 EGEs in day.csv"):
        check_totals_kept(
            pd.Series({"a": 3.0, "b": 5.0}), pd.Series({"a": 2.0}), "day.csv"
        )


@pytest.mark.parametrize(
    "zarr_ids, stac_ids, errors",
    [
        ({"a", "b"}, ["a", "b"], []),
        ({"a", "b"}, ["a"], ["1 EGE IDs have no STAC data, e.g. ['b']"]),
        ({"a"}, ["a", "c"], ["1 EGE IDs have no Zarr data, e.g. ['c']"]),
    ],
)
def test_log_unjoined_ids(
    zarr_ids: set[str], stac_ids: list[str], errors: list[str]
) -> None:
    """Happy & failure path: an ID only one of the Zarr and STAC data has is logged.

    Args:
        zarr_ids (set[str]): The distinct IDs in the Zarr data.
        stac_ids (list[str]): The IDs in the STAC data.
        errors (list[str]): The expected error messages.
    """
    captured_logs: list[str] = []
    sink_id = logger.add(lambda msg: captured_logs.append(msg.record["message"]))
    try:
        log_unjoined_ids(zarr_ids, pd.Series(stac_ids))
    finally:
        logger.remove(sink_id)

    assert captured_logs == errors


# ----------------------------------
# Tests: base processor
# ----------------------------------
class _Processor(EnergyProcessor):
    """The least a processor needs: its class attributes and the two required hooks."""

    DATASET = MagicMock(sysop="aemo")  # not name_matchable: no coordinate finding
    TZ = "America/Sao_Paulo"
    COLUMNS = {}
    EGE_COLUMNS = []

    def _prepare_for_zarr(self, df_eg, period):  # noqa: D102
        return df_eg, {}

    def _get_stac_fields(self, df_ege_locators, df_ege_sysop, active_periods, totals):  # noqa: D102
        return pd.DataFrame(index=pd.Index([], name="id"))


def _make(cls: type[EnergyProcessor], tmp_path: Path) -> EnergyProcessor:
    """Build a processor that writes into a temporary directory and doesn't publish.

    Args:
        cls (type[EnergyProcessor]): The processor class.
        tmp_path (Path): Pytest-provided temporary directory.

    Returns:
        EnergyProcessor: The processor.
    """
    return cls(
        input_dir=Path(tmp_path, "raw"),
        resources_dir=Path(tmp_path, "resources"),
        output_dir=Path(tmp_path, "processed"),
        release="v0.1.0",
        publish_cfg=MagicMock(),  # not publishing: never read
        publish=False,
    )


def test_processor_without_its_class_attributes_raises(tmp_path: Path) -> None:
    """Failure path: a subclass that doesn't set all class attributes can't be built.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.
    """

    class _Incomplete(_Processor):
        pass

    del _Processor.COLUMNS  # temporarily, as if never set
    try:
        with pytest.raises(TypeError, match="COLUMNS"):
            _make(_Incomplete, tmp_path)
    finally:
        _Processor.COLUMNS = {}


def test_processor_without_coordinate_finding_skips_its_resources(
    tmp_path: Path,
) -> None:
    """Happy path: without coordinate finding, its expensive resources aren't built.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.
    """
    with patch.object(processing, "build_shared_resources") as mock_build:
        proc = _make(_Processor, tmp_path)
    mock_build.assert_not_called()
    assert proc.shared is None


def test_review_files_written_only_with_rows(tmp_path: Path) -> None:
    """Happy path: each review file gets its rows from all files, empty ones none.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.
    """
    proc = _make(_Processor, tmp_path)
    rows = pd.DataFrame({"id": ["a"], "value": [1.0]})
    proc._write_review(
        {"repeated_hour_rows": [rows, rows], "skipped": [rows.iloc[:0]]}, run="1h"
    )

    review = Path(tmp_path, "processed", "review")
    assert [f.name for f in review.iterdir()] == ["repeated_hour_rows_1h.csv"]
    assert len(pd.read_csv(Path(review, "repeated_hour_rows_1h.csv"))) == 2


def test_stac_fields_with_an_ege_twice_raise(tmp_path: Path) -> None:
    """Failure path: fields giving an EGE two rows raise, rather than two Items one ID.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.
    """

    class _Twice(_Processor):
        def _get_stac_fields(
            self, df_ege_locators, df_ege_sysop, active_periods, totals
        ):  # noqa: D102
            return pd.DataFrame({"fuel_type": ["solar", "wind"]}, index=["a", "a"])

    df_ege_locators = pd.DataFrame({"id": ["a"], processing.SYSOP_NAME_COL: ["SOL A"]})
    empty = pd.Series(dtype=object)
    with pytest.raises(ValueError, match=r"1 EGEs more than one row, e.g. \['a'\]"):
        _make(_Twice, tmp_path)._prepare_for_stac(
            df_ege_locators, pd.DataFrame(), empty, empty
        )
