# tests/publish/conftest.py
from functools import cache

import pytest
import requests


@cache
def _online() -> bool:
    """Tell whether the official STAC schemas can be fetched (once per test run).

    Returns:
        bool: True if schemas.stacspec.org answers.
    """
    try:
        requests.head("https://schemas.stacspec.org/", timeout=5)
    except requests.RequestException:
        return False
    return True


@pytest.fixture
def needs_stac_schemas() -> None:
    """Skip a test that validates STAC fully when the official schemas can't be fetched."""
    if not _online():
        pytest.skip("STAC schemas can't be fetched (offline).")
