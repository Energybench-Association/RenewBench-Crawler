# tests/publish/test_stac.py
"""Tests for a release's STAC Collection and Items."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from rbc.publish import stac
from rbc.publish.icechunk_store import StorageInfo

INFO = stac.DatasetInfo(
    dataset="op",
    title="Operator",
    description="Test data.",
    license="proprietary",
    license_url="https://example.org/terms",
    providers=[{"name": "Operator", "roles": ["producer", "licensor"]}],
    citation="Operator: generation data.",
    values="Hourly gross generation (MW)",
)
ENTITIES = ["op_a", "op_b"]
TIMES = pd.date_range("2024-01-01 03:00", "2024-03-01 02:00", freq="h")
STORAGE = StorageInfo(
    prefix="energy/op/PT1H/repository",
    bucket="energybench",
    data_base_url="https://data.example.org",
)
POINT = {"type": "Point", "coordinates": [-70.0, -33.0]}


def _rows(**overrides: list) -> pd.DataFrame:
    """Two EGEs' STAC data.

    Args:
        **overrides (list): Columns to replace.

    Returns:
        pd.DataFrame: The rows.
    """
    rows = {
        "id": ["op_a", "op_b"],
        "geometry": [POINT, POINT],
        "entity_name": ["A", "B"],
        "entity_type": ["facility", "facility"],
        "fuel_type": ["wind", "hybrid"],
        "hybrid_fuel_types": [None, ["gas", "oil"]],
        "generation_capacity": [10.0, float("nan")],
        "entity_ids": ['{"op": "1"}', None],
        "measurement_type": ["gross_output", "gross_output"],
    }
    return pd.DataFrame({**rows, **overrides})


SPANS = pd.DataFrame(
    {
        "start": pd.to_datetime(["2024-01-01 03:00", "2024-01-02 03:00"], utc=True),
        "end": pd.to_datetime(["2024-02-01 03:00", "2024-03-01 03:00"], utc=True),
    },
    index=["op_a", "op_b"],
)


def _built(df: pd.DataFrame | None = None) -> tuple[dict, list[dict]]:
    """Build a release's Collection and Items.

    Args:
        df (pd.DataFrame | None, optional): The STAC data. Defaults to `_rows()`.

    Returns:
        tuple[dict, list[dict]]: The Collection and the Items.
    """
    items = stac.build_items(
        _rows() if df is None else df, SPANS, INFO, "v0.1.0", STORAGE, "SNAP"
    )
    coll = stac.build_collection(
        INFO,
        "v0.1.0",
        items,
        STORAGE,
        "SNAP",
        entities=ENTITIES,
        times=TIMES,
        step="PT1H",
    )
    return coll, items


def test_items_and_collection_are_valid(needs_stac_schemas: None) -> None:
    """Happy path: the records are valid STAC (incl. every extension they use).

    Args:
        needs_stac_schemas (None): Skips the test when the schemas can't be fetched.
    """
    coll, items = _built()
    stac.validate(coll, items)


def test_items_carry_the_energy_fields_and_data_asset() -> None:
    """Happy path: Items hold the energy fields and the repository's data asset."""
    coll, items = _built()

    a, b = items
    assert a["properties"]["energy:entity_ids"] == {"op": "1"}
    assert "energy:generation_capacity" not in b["properties"]  # NaN left out
    assert b["properties"]["energy:hybrid_fuel_types"] == ["gas", "oil"]
    assert a["properties"]["start_datetime"] == "2024-01-01T03:00:00Z"
    assert a["assets"]["data"] == {
        "href": "https://data.example.org/energybench/energy/op/PT1H/repository",
        "type": "application/vnd.icechunk",
        "title": "Icechunk repository @ v0.1.0",
        "roles": ["data"],
        "icechunk:repository": "energy/op/PT1H/repository",
        "icechunk:tag": "v0.1.0",
        "icechunk:snapshot": "SNAP",
        "description": "Hourly gross generation (MW) of this entity: 'values' at "
        "entity_id 'op_a', time in UTC.",
        "icechunk:entity_id": "op_a",
    }
    assert {link["rel"] for link in a["links"]} == {"collection", "parent", "root"}
    assert a["properties"]["created"].endswith("Z")
    assert coll["id"] == "op__v0.1.0" == a["collection"]
    assert coll["summaries"]["energy:fuel_type"] == ["hybrid", "wind"]
    assert coll["summaries"]["energy:generation_capacity"] == {
        "minimum": 10.0,
        "maximum": 10.0,
    }
    assert coll["extent"]["temporal"]["interval"] == [
        ["2024-01-01T03:00:00Z", "2024-03-01T03:00:00Z"]
    ]
    assert coll["themes"][0]["concepts"] == [{"id": "generation"}]


def test_collection_describes_the_repository() -> None:
    """Happy path: the Collection holds the repository, its array and how to cite it."""
    coll, items = _built()

    assert coll["assets"]["data"]["icechunk:snapshot"] == "SNAP"
    assert "icechunk:entity_id" not in coll["assets"]["data"]
    assert coll["item_assets"]["data"]["type"] == "application/vnd.icechunk"
    assert coll["cube:dimensions"]["entity_id"]["values"] == ENTITIES
    assert coll["cube:dimensions"]["time"]["extent"] == [
        "2024-01-01T03:00:00Z",
        "2024-03-01T02:00:00Z",
    ]
    assert coll["cube:variables"]["values"]["unit"] == "MW"
    assert coll["sci:citation"] == INFO.citation
    item_links = [link["href"] for link in coll["links"] if link["rel"] == "item"]
    assert item_links == ["./items/op_a.json", "./items/op_b.json"]


def test_validate_rejects_a_missing_required_field(needs_stac_schemas: None) -> None:
    """Failure path: an Item without a fuel type breaks the energy extension.

    Args:
        needs_stac_schemas (None): Skips the test when the schemas can't be fetched.
    """
    coll, items = _built()
    del items[0]["properties"]["energy:fuel_type"]
    with pytest.raises(ValueError, match="op_a"):
        stac.validate(coll, items)


def test_ege_without_values_gets_no_item() -> None:
    """Happy path: an EGE with no value at all has no span, so no Item."""
    df = _rows(id=["op_a", "op_c"])
    items = stac.build_items(df, SPANS, INFO, "v0.1.0", STORAGE, "SNAP")
    assert [it["id"] for it in items] == ["op_a"]


def test_ege_without_geometry_raises() -> None:
    """Failure path: an EGE that can't be placed stops building."""
    with pytest.raises(ValueError, match="without geometry"):
        _built(_rows(geometry=[POINT, None]))


def test_write_stac(tmp_path: Path) -> None:
    """Happy path: one folder per Collection, one JSON file per Item."""
    coll, items = _built()
    folder = stac.write_stac(tmp_path, coll, items)
    assert json.loads(Path(folder, "collection.json").read_text())["id"] == coll["id"]
    assert sorted(p.stem for p in Path(folder, "items").iterdir()) == ["op_a", "op_b"]


def test_publish_creates_or_replaces() -> None:
    """Happy path: existing records (409) are replaced; a refusal raises."""
    coll, items = _built()
    http = MagicMock()
    http.post.side_effect = [
        MagicMock(status_code=201, ok=True),
        MagicMock(status_code=409, ok=False),
        MagicMock(status_code=201, ok=True),
    ]
    http.put.return_value = MagicMock(status_code=200, ok=True)

    with patch.object(stac.requests, "Session", return_value=http):
        stac.publish("https://stac.example.org/", coll, items)

    assert http.post.call_args_list[0].args[0] == "https://stac.example.org/collections"
    # the static catalog's relative links stay out: the API links records itself
    sent = [c.kwargs["json"] for c in http.post.call_args_list]
    assert [link["rel"] for link in sent[0]["links"]] == ["license"]
    assert sent[1]["links"] == []
    http.put.assert_called_once()
    assert (
        http.put.call_args.args[0]
        == "https://stac.example.org/collections/op__v0.1.0/items/op_a"
    )

    http.post.side_effect = [MagicMock(status_code=400, ok=False, text="bad")]
    with patch.object(stac.requests, "Session", return_value=http):
        with pytest.raises(stac.requests.HTTPError, match="bad"):
            stac.publish("https://stac.example.org", coll, items)
