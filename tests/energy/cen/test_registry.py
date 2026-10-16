# tests/energy/cen/test_registry.py
"""Tests for CEN's EGE registry: the API download, its cache and the UTM conversion."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from requests import exceptions

from rbc.energy.cen import registry
from rbc.energy.cen.registry import CenCoordinateRegistry, add_latlon, utm_to_latlon
from rbc.energy.utils import MAX_RETRIES, DataStructureError

# a point in Antofagasta (zone 19K) and one in O'Higgins (zone 19H), as CEN formats them
ANTOFAGASTA = ("300.000", "7.460.000", "19 K")
ANTOFAGASTA_LATLON = (-22.9562, -70.9507)
OHIGGINS = ("679310,49", "6.287.533,3", "19H")
OHIGGINS_LATLON = (-33.5368, -67.0689)


# ----------------------------------
# Fixtures
# ----------------------------------
def _api_row(id_central: int, east: str | None, north: str | None, zone: str) -> dict:
    """One EGE as the registry API returns it.

    Args:
        id_central (int): The EGE's ID.
        east (str | None): Its UTM easting, as CEN formats it.
        north (str | None): Its UTM northing, as CEN formats it.
        zone (str): Its UTM zone.

    Returns:
        dict: The API row.
    """
    return {
        "id_central": id_central,
        "central": f"PFV {id_central}",
        "tipo_tecnologia": "Solar",
        "estado": "Operativa",
        "comuna": "Taltal",
        "region": "Antofagasta",
        "coordenada_este": east,
        "coordenada_norte": north,
        "zona_huso": zone,
    }


def _response(rows: list[dict], total_pages: int = 1) -> MagicMock:
    """A successful API response holding one page of rows.

    Args:
        rows (list[dict]): The page's rows.
        total_pages (int): How many pages the API reports in total.

    Returns:
        MagicMock: Stand-in for a `requests.Response`.
    """
    response = MagicMock()
    response.json.return_value = {"data": rows, "totalPages": total_pages}
    return response


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the waits between retries.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest-provided monkeypatch fixture.
    """
    monkeypatch.setattr(registry.time, "sleep", lambda seconds: None)


# ----------------------------------
# Tests: UTM conversion
# ----------------------------------
@pytest.mark.parametrize(
    "east, north, zone, expected",
    [
        (*ANTOFAGASTA, ANTOFAGASTA_LATLON),  # dots grouping thousands
        ("300000", "7460000", "19K", ANTOFAGASTA_LATLON),  # plain digits
        ("300000.0", "7460000.0", "19", ANTOFAGASTA_LATLON),  # decimal point
        (*OHIGGINS, OHIGGINS_LATLON),  # decimal comma (with grouping dots)
    ],
)
def test_utm_formats_convert(
    east: str, north: str, zone: str, expected: tuple[float, float]
) -> None:
    """Happy path: each of CEN's number formats converts to the same point.

    CEN writes '.' both as a thousands separator and as a decimal point, so the format
    decides the value: '300.000' is 300 km, '5762380.85' a decimal.

    Args:
        east (str): UTM easting, as CEN formats it.
        north (str): UTM northing, as CEN formats it.
        zone (str): UTM zone.
        expected (tuple[float, float]): The expected (lat, lon).
    """
    assert utm_to_latlon(east, north, zone) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize(
    "east, north, zone",
    [
        ("x", "7460000", "19K"),  # unparsable number
        (None, "7460000", "19K"),  # missing number
        (float("nan"), "7460000", "19K"),  # missing number, as read from the cache
        ("300000", "7460000", "nozone"),  # no zone number
        ("300000", "7460000", "99"),  # no such zone
        ("300000", "7460000", "12"),  # converts, but lands far west of Chile
    ],
)
def test_utm_without_a_valid_point(
    east: str | float | None, north: str, zone: str
) -> None:
    """Failure path: a value that isn't a point in Chile gives no coordinates at all.

    Args:
        east (str | float | None): UTM easting.
        north (str): UTM northing.
        zone (str): UTM zone.
    """
    assert utm_to_latlon(east, north, zone) == (None, None)


def test_add_latlon_keeps_unconvertible_rows() -> None:
    """Happy path: every registry row stays, with coordinates only where convertible."""
    df = pd.DataFrame([_api_row(1, *ANTOFAGASTA), _api_row(2, None, "7460000", "19K")])

    out = add_latlon(df)

    assert len(out) == 2
    assert (out.loc[0, "lat"], out.loc[0, "lon"]) == pytest.approx(
        ANTOFAGASTA_LATLON, abs=1e-4
    )
    assert out.loc[1, ["lat", "lon"]].isna().all()


# ----------------------------------
# Tests: download & cache
# ----------------------------------
def test_download_is_cached_as_the_api_delivers_it(tmp_path: Path) -> None:
    """Happy path: the cache holds the API's own fields, the registry adds lat/lon.

    Keeping the cache to what the API delivers means a fix in the conversion applies on
    the next load, without downloading the registry again.

    Args:
        tmp_path (Path): Pytest-provided temporary directory (the cache dir).
    """
    with patch.object(
        registry.requests, "get", return_value=_response([_api_row(1, *ANTOFAGASTA)])
    ):
        reg = CenCoordinateRegistry(token="token", cache_dir=tmp_path)

    cached = pd.read_csv(next(tmp_path.glob("centrales_*.csv")))
    assert "lat" not in cached.columns
    assert {"coordenada_este", "coordenada_norte", "zona_huso"} <= set(cached.columns)
    assert (reg.df_registry.loc[0, "lat"], reg.df_registry.loc[0, "lon"]) == (
        pytest.approx(ANTOFAGASTA_LATLON, abs=1e-4)
    )


def test_cache_reads_utm_fields_as_text(tmp_path: Path) -> None:
    """Happy path: a cached registry converts exactly like a freshly downloaded one.

    Read as numbers, a column of only number-like values turns '300.000' (300 km) into
    300.0 (300 m), and the point lands outside Chile.

    Args:
        tmp_path (Path): Pytest-provided temporary directory (the cache dir).
    """
    rows = [_api_row(1, *ANTOFAGASTA)]  # one row: the easting column is all number-like
    with patch.object(registry.requests, "get", return_value=_response(rows)):
        fresh = CenCoordinateRegistry(token="token", cache_dir=tmp_path).df_registry

    with patch.object(registry.requests, "get", side_effect=AssertionError("no API")):
        cached = CenCoordinateRegistry(token="token", cache_dir=tmp_path).df_registry

    pd.testing.assert_frame_equal(cached[["lat", "lon"]], fresh[["lat", "lon"]])
    assert cached[["lat", "lon"]].notna().all().all()


def test_download_collects_every_page(tmp_path: Path) -> None:
    """Happy path: rows from every page are collected, each EGE once.

    Args:
        tmp_path (Path): Pytest-provided temporary directory (the cache dir).
    """
    pages = [
        _response([_api_row(1, *ANTOFAGASTA), _api_row(2, *OHIGGINS)], total_pages=2),
        _response([_api_row(2, *OHIGGINS), _api_row(3, *ANTOFAGASTA)], total_pages=2),
    ]
    with patch.object(registry.requests, "get", side_effect=pages):
        reg = CenCoordinateRegistry(token="token", cache_dir=tmp_path)

    assert sorted(reg.df_registry["id_central"]) == [1, 2, 3]


def test_download_retries_a_failed_request(tmp_path: Path) -> None:
    """Happy path: a failed request is retried rather than ending the run.

    Args:
        tmp_path (Path): Pytest-provided temporary directory (the cache dir).
    """
    attempts = [exceptions.ConnectionError(), _response([_api_row(1, *ANTOFAGASTA)])]
    with patch.object(registry.requests, "get", side_effect=attempts) as mock_get:
        reg = CenCoordinateRegistry(token="token", cache_dir=tmp_path)

    assert mock_get.call_count == 2
    assert len(reg.df_registry) == 1


def test_download_gives_up_after_all_retries(tmp_path: Path) -> None:
    """Failure path: a request still failing after every retry raises.

    Args:
        tmp_path (Path): Pytest-provided temporary directory (the cache dir).
    """
    with patch.object(
        registry.requests, "get", side_effect=exceptions.ConnectionError()
    ) as mock_get:
        with pytest.raises(exceptions.ConnectionError):
            CenCoordinateRegistry(token="token", cache_dir=tmp_path)

    assert mock_get.call_count == MAX_RETRIES + 1
    assert not list(tmp_path.glob("*.csv"))  # nothing cached


@pytest.mark.parametrize(
    "body",
    [
        {"rows": [], "totalPages": 1},  # no "data" field
        {"data": [{"central": "PFV 1"}], "totalPages": 1},  # an EGE without its ID
    ],
)
def test_changed_structure_raises(tmp_path: Path, body: dict) -> None:
    """Failure path: a changed API structure ends the run, rather than caching garbage.

    Args:
        tmp_path (Path): Pytest-provided temporary directory (the cache dir).
        body (dict): The changed response body.
    """
    response = MagicMock()
    response.json.return_value = body
    with patch.object(registry.requests, "get", return_value=response):
        with pytest.raises(DataStructureError):
            CenCoordinateRegistry(token="token", cache_dir=tmp_path)
