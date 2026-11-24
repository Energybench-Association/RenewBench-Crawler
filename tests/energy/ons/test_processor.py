# tests/energy/ons/test_processor.py
"""Tests for the ONS processor's operator-specific steps (the shared ones: s. utils)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from rbc.energy.ons.processor import OnsProcessor
from rbc.energy.utils import processing


@pytest.fixture
def proc(tmp_path: Path) -> OnsProcessor:
    """An ONS processor without coordinate finding's resources, not publishing.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.

    Returns:
        OnsProcessor: The processor.
    """
    with patch.object(processing, "build_shared_resources", return_value=MagicMock()):
        return OnsProcessor(
            input_dir=Path(tmp_path, "raw", "ons"),
            resources_dir=Path(tmp_path, "resources"),
            output_dir=Path(tmp_path, "processed", "ons"),
            release="v0.1.0",
            publish_cfg=MagicMock(),  # not publishing: never read
            publish=False,
        )


def _rows(times: list[str], ids: list[str] | None = None) -> pd.DataFrame:
    """A raw CSV's rows as `_prepare_for_zarr` gets them (layout names, with "id").

    Args:
        times (list[str]): The rows' local 'din_instante'.
        ids (list[str] | None, optional): The rows' EGE IDs. Defaults to one EGE.

    Returns:
        pd.DataFrame: The rows, 1 MW each.
    """
    return pd.DataFrame(
        {"id": ids or ["ons-a"] * len(times), "timestamp": times, "value": 1.0}
    )


# ----------------------------------
# Tests: raw files
# ----------------------------------
def test_unknown_or_missing_mode_raises(proc: OnsProcessor) -> None:
    """Failure path: a mode the processor doesn't classify stops processing.

    Args:
        proc (OnsProcessor): The processor.
    """
    df = pd.DataFrame({"cod_modalidadeoperacao": ["TIPO I", "TIPO IV", None]})
    with pytest.raises(ValueError, match=r"\['<missing>', 'TIPO IV'\]"):
        proc._clean_file(df, "2025-07")


def test_clocks_back_hour_is_left_out_for_review(proc: OnsProcessor) -> None:
    """Happy path: the repeated 23:00 (one row, two real hours) goes to review.

    On 2018-02-17, Brazil's clocks went from 24:00 (UTC-2) back to 23:00 (UTC-3).

    Args:
        proc (OnsProcessor): The processor.
    """
    df = _rows(["2018-02-17 22:00", "2018-02-17 23:00", "2018-02-18 00:00"])
    out, review = proc._prepare_for_zarr(df, "2018-02")

    assert out["timestamp"].tolist() == [
        pd.Timestamp("2018-02-18 00:00", tz="UTC"),  # 22:00 DST
        pd.Timestamp("2018-02-18 03:00", tz="UTC"),  # 00:00 standard time
    ]
    assert review["repeated_hour_rows"]["timestamp"].tolist() == ["2018-02-17 23:00"]


@pytest.mark.parametrize(
    "times, month, match",
    [
        (["2018-03-01 00:00"], "2018-02", "outside the month"),
        (["2018-02-01 00:30"], "2018-02", "not on the hour"),
        (["2018-11-04 00:00"], "2018-11", "DST change skips"),  # clocks forward
    ],
)
def test_hours_that_cant_be_placed_raise(
    proc: OnsProcessor, times: list[str], month: str, match: str
) -> None:
    """Failure path: hours outside the file's month, off the hour or skipped raise.

    Args:
        proc (OnsProcessor): The processor.
        times (list[str]): The rows' local 'din_instante'.
        month (str): The month the file holds.
        match (str): The expected error message part.
    """
    with pytest.raises(ValueError, match=match):
        proc._prepare_for_zarr(_rows(times), month)


def test_hour_reported_twice_raises(proc: OnsProcessor) -> None:
    """Failure path: one EGE's hour in two rows raises, rather than picking one.

    Args:
        proc (OnsProcessor): The processor.
    """
    with pytest.raises(ValueError, match="reported twice"):
        proc._prepare_for_zarr(_rows(["2020-01-01 00:00"] * 2), "2020-01")


# ----------------------------------
# Tests: STAC fields
# ----------------------------------
def test_stac_fuels_hybrid_and_unknown() -> None:
    """Happy and failure path: two technologies make a hybrid, unmapped ones raise."""
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "fuel_type": ["EOLIELÉTRICA", "TÉRMICA"],
            "fuel_subtype": ["Fotovoltaica", "Óleo Diesel"],
        }
    )
    out = OnsProcessor._stac_fuels(df)
    assert out.values.tolist() == [
        ["a", "wind", "onshore"],
        ["a", "solar", "pv"],
        ["b", "oil", "diesel"],
    ]

    df.loc[1, "fuel_subtype"] = "Unknown fuel"
    with pytest.raises(ValueError, match="Unknown fuel"):
        OnsProcessor._stac_fuels(df)


def test_ege_fields_by_operating_mode(proc: OnsProcessor) -> None:
    """Happy path: entity and measurement types follow the mode, parts get a facility.

    Args:
        proc (OnsProcessor): The processor.
    """
    raw = [  # id, name, ceg, mode, located
        ("ons-c", "Conj. A", "-", "Conjunto de Usinas", True),
        ("ons-p", "Pequenas Usinas X", "-", "Pequenas Usinas (Tipo III)", False),
        ("ons-m", "PQU X FOTOV MMGD", "-", "Pequenas Usinas (MMGD)", False),
        ("ons-f1", "Flores 1", "UTE.PE.AM.029192-7.01", "TIPO III", False),
        ("ons-f2", "Flores 2", "UTE.PE.AM.029192-7.01", "TIPO III", False),
        ("ons-i", "Itaipu 60 Hz", "UHE.PH.PR.001161-4.01", "TIPO I", True),
    ]
    ids = [r[0] for r in raw]
    df_ege_locators = pd.DataFrame(
        {
            "id": ids,
            "entity_name": [r[1] for r in raw],
            "lat": [-3.0 if r[4] else None for r in raw],
            "lon": [-60.0 if r[4] else None for r in raw],
        }
    )
    df_ege_sysop = pd.DataFrame(
        {
            "id": ids,
            "id_ons": ids,
            "ceg": [r[2] for r in raw],
            "cod_modalidadeoperacao": [r[3] for r in raw],
            "fuel_type": "TÉRMICA",
            "fuel_subtype": "Óleo Diesel",
        }
    )
    active_periods = pd.Series({i: {"2020-01"} for i in ids}, dtype=object)
    totals = pd.Series(1.0, index=ids)

    out = proc._get_stac_fields(df_ege_locators, df_ege_sysop, active_periods, totals)

    assert out["entity_type"].to_dict() == {
        "ons-c": "facility",  # a group at one point
        "ons-p": "portfolio",  # a group without one
        "ons-m": "portfolio",
        "ons-f1": "group",  # parts of one facility (shared CEG)
        "ons-f2": "group",
        "ons-i": "facility",
    }
    assert out.loc["ons-f1", "facility_name"] == "Flores"
    assert out["measurement_type"].to_dict() == {
        "ons-c": "gross_output",
        "ons-p": "forecast",
        "ons-m": "unknown",
        "ons-f1": "forecast",  # Tipo III
        "ons-f2": "forecast",
        "ons-i": "gross_output",
    }
    assert out["ceg"].isna().tolist() == [True, True, True, False, False, False]
