# tests/publish/test_stac.py
"""Tests for a release's STAC Collection and Items."""

import json
import subprocess
import tomllib
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pystac.errors import STACValidationError

from rbc.publish import stac
from rbc.publish.icechunk_store import StorageInfo

INFO = stac.DatasetInfo(
    sysop="op",
    title="Operator {tres_verbose} generation",
    description="{Tres_verbose} test data.",
    license="proprietary",
    license_url="https://example.org/terms",
    providers=[
        {"name": "Operator", "roles": ["producer", "licensor"]},
        {"name": "Processor", "roles": ["processor", "host"]},
    ],
    citation="Operator: generation data.",
    lineage="{Tres_verbose} values converted to UTC.",
    values="{Tres_verbose} gross generation (MW)",
    sysop_long="Grid Operator",
    countries={"CL": "Chile", "AR": "Argentina"},
    continents=["South America"],
    keywords=["Somewhere"],
)
ENTITIES = ["op_a", "op_b"]
TIMES = pd.date_range("2024-01-01 03:00", "2024-03-01 02:00", freq="h")
STORAGE = StorageInfo(
    prefix="energy/op/PT1H/repository",
    bucket="energybench",
    url_public="https://data.example.org",
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
    builder = stac.EnergyStacBuilder(INFO, "v0.1.0", STORAGE, "SNAP", "PT1H")
    items = builder.build_items(_rows() if df is None else df, SPANS)
    coll = builder.build_collection(items, entities=ENTITIES, times=TIMES)
    return coll, items


def test_validate_checks_every_record() -> None:
    """Happy path: the Collection and each Item are validated, without network access."""
    coll, items = _built()
    with patch.object(stac, "validate_dict") as mock_validate:
        stac.validate(coll, items)

    checked = [c.args[0]["id"] for c in mock_validate.call_args_list]
    assert checked == [coll["id"], "op_a", "op_b"]


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
    assert coll["id"] == "op-pt1h__v0.1.0" == a["collection"]
    assert coll["summaries"]["energy:fuel_type"] == ["hybrid", "wind"]
    assert coll["summaries"]["energy:generation_capacity"] == {
        "minimum": 10.0,
        "maximum": 10.0,
    }
    assert coll["extent"]["temporal"]["interval"] == [
        ["2024-01-01T03:00:00Z", "2024-03-01T03:00:00Z"]
    ]
    assert coll["themes"] == [
        {
            "scheme": "https://energybench.org/domains",
            "concepts": [{"id": "generation"}],
        },
        {
            "scheme": "https://energybench.org/system-operators",
            "concepts": [{"id": "op", "title": "Grid Operator"}],
        },
        {
            "scheme": "https://energybench.org/countries",
            "concepts": [
                {"id": "CL", "title": "Chile"},
                {"id": "AR", "title": "Argentina"},
            ],
        },
        {
            "scheme": "https://energybench.org/continents",
            "concepts": [{"id": "south-america", "title": "South America"}],
        },
    ]


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


def test_collection_states_the_lineage() -> None:
    """Happy path: the processor carries the lineage, which closes the description."""
    coll, _ = _built()

    operator, processor = coll["providers"]
    assert "processing:lineage" not in operator
    assert processor["processing:lineage"] == "Hourly values converted to UTC."
    assert coll["description"] == "Hourly test data.\n\nHourly values converted to UTC."
    assert processor["processing:software"] == stac.get_software()
    assert stac.PROCESSING_EXT in coll["stac_extensions"]


def test_software_is_the_projects_from_pyproject() -> None:
    """Happy path: the software is named and versioned as the repo's pyproject.toml has it."""
    stac.get_software.cache_clear()
    (name, version_commit) = stac.get_software().popitem()
    project = tomllib.loads(stac.PYPROJECT.read_text())["project"]
    assert name == project["name"]
    assert version_commit.split("+g")[0] == project["version"]
    stac.get_software.cache_clear()


@pytest.mark.parametrize(
    "git, expected",
    [
        (MagicMock(stdout="4176167\n"), "0.0.1+g4176167"),
        (subprocess.CalledProcessError(128, "git"), "0.0.1"),  # not a git checkout
        (FileNotFoundError("git"), "0.0.1"),  # no git installed
    ],
)
def test_software_version_adds_the_commit(
    tmp_path: Path, git: object, expected: str
) -> None:
    """Happy path: the project's version gets the git commit, where there is one.

    Args:
        tmp_path (Path): Pytest's temporary directory, for a pyproject.toml.
        git (object): What running git gives: its output, or the error it raises.
        expected (str): The version expected.
    """
    pyproject = Path(tmp_path, "pyproject.toml")
    pyproject.write_text('[project]\nname = "Some-Crawler"\nversion = "0.0.1"\n')
    stac.get_software.cache_clear()
    run = {"side_effect": git} if isinstance(git, Exception) else {"return_value": git}
    with (
        patch.object(stac, "PYPROJECT", pyproject),
        patch.object(stac.subprocess, "run", **run),
    ):
        assert stac.get_software() == {"Some-Crawler": expected}
    stac.get_software.cache_clear()


def test_software_without_pyproject_is_unknown(tmp_path: Path) -> None:
    """Failure path: without pyproject.toml, the import package is named, version unknown.

    Args:
        tmp_path (Path): Pytest's temporary directory, without a pyproject.toml.
    """
    stac.get_software.cache_clear()
    with (
        patch.object(stac, "PYPROJECT", Path(tmp_path, "pyproject.toml")),
        patch.object(stac.subprocess, "run", side_effect=FileNotFoundError("git")),
    ):
        assert stac.get_software() == {"rbc": "unknown"}
    stac.get_software.cache_clear()


def test_collection_without_processor_raises() -> None:
    """Failure path: without a processor provider, the lineage has nowhere to go."""
    info = replace(INFO, providers=INFO.providers[:1])
    with pytest.raises(ValueError, match="processor"):
        stac.EnergyStacBuilder(
            info, "v0.1.0", STORAGE, "SNAP", "PT1H"
        ).build_collection(_built()[1], ENTITIES, TIMES)


@pytest.mark.parametrize(
    "notice, licenses",
    [
        (None, ["https://example.org/terms"]),
        (
            "https://example.org/data#notice",
            ["https://example.org/terms", "https://example.org/data#notice"],
        ),
    ],
)
def test_collection_links_the_license_notice(
    notice: str | None, licenses: list[str]
) -> None:
    """Happy path: a source's license notice, if any, is a second license link.

    Args:
        notice (str | None): Where the source states its license.
        licenses (list[str]): The Collection's license links expected.
    """
    info = replace(INFO, license_notice_url=notice)
    builder = stac.EnergyStacBuilder(info, "v0.1.0", STORAGE, "SNAP", "PT1H")
    coll = builder.build_collection(_built()[1], ENTITIES, TIMES)

    assert [ln["href"] for ln in coll["links"] if ln["rel"] == "license"] == licenses


@pytest.mark.parametrize(
    "tres_iso, tres_verbose",
    [("PT1H", "hourly"), ("PT15M", "15-minute"), ("P1D", "daily"), ("PT3H", "3-hour")],
)
def test_texts_name_the_resolution(tres_iso: str, tres_verbose: str) -> None:
    """Happy path: the dataset's texts and keywords name the resolution in words.

    Args:
        tres_iso (str): The resolution as an ISO 8601 duration.
        tres_verbose (str): How the texts name it.
    """
    info = INFO.for_resolution(tres_iso)
    assert info.title == f"Operator {tres_verbose} generation"
    assert info.description == f"{tres_verbose.capitalize()} test data."
    assert info.values == f"{tres_verbose.capitalize()} gross generation (MW)"
    assert info.keywords == ["Somewhere", tres_verbose]
    cid = stac._collection_id("op", tres_iso, "v0.1.0")
    assert cid == f"op-{tres_iso.lower()}__v0.1.0"


def test_texts_reject_other_durations() -> None:
    """Failure path: a month (ISO 8601 "P1M") isn't a resolution the texts can name."""
    with pytest.raises(ValueError, match="P1M"):
        INFO.for_resolution("P1M")


def test_validate_names_the_invalid_record() -> None:
    """Failure path: an invalid Item raises with its ID and the schema's reason."""
    coll, items = _built()

    def _reject_op_a(obj: dict) -> None:
        if obj["id"] == "op_a":
            raise STACValidationError(
                "Validation failed against schema at https://example.org/energy.json\n"
                "'energy:fuel_type' is a required property"
            )

    with (
        patch.object(stac, "validate_dict", side_effect=_reject_op_a),
        pytest.raises(
            ValueError,
            match="STAC Feature 'op_a' is invalid: 'energy:fuel_type' is a required",
        ),
    ):
        stac.validate(coll, items)


def test_ege_without_values_gets_no_item() -> None:
    """Happy path: an EGE with no value at all has no span, so no Item."""
    df = _rows(id=["op_a", "op_c"])
    builder = stac.EnergyStacBuilder(INFO, "v0.1.0", STORAGE, "SNAP", "PT1H")
    items = builder.build_items(df, SPANS)
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


def test_drop_stac_replaces_a_waiting_release(tmp_path: Path) -> None:
    """Happy path: a release is moved into the inbox whole, replacing one still waiting.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.
    """
    coll, items = _built()
    stale = Path(tmp_path, coll["id"], "items", "gone.json")
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")

    folder = stac.drop_stac(tmp_path, coll, items)

    assert folder == Path(tmp_path, coll["id"])
    assert sorted(p.stem for p in Path(folder, "items").iterdir()) == ["op_a", "op_b"]
    assert not any(Path(tmp_path, ".incoming").iterdir())


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
        stac.publish_stac("https://stac.example.org/", coll, items)

    assert http.post.call_args_list[0].args[0] == "https://stac.example.org/collections"
    # the static catalog's relative links stay out: the API links records itself
    sent = [c.kwargs["json"] for c in http.post.call_args_list]
    assert [link["rel"] for link in sent[0]["links"]] == ["license"]
    assert sent[1]["links"] == []
    http.put.assert_called_once()
    assert (
        http.put.call_args.args[0]
        == "https://stac.example.org/collections/op-pt1h__v0.1.0/items/op_a"
    )

    http.post.side_effect = [MagicMock(status_code=400, ok=False, text="bad")]
    with patch.object(stac.requests, "Session", return_value=http):
        with pytest.raises(stac.requests.HTTPError, match="bad"):
            stac.publish_stac("https://stac.example.org", coll, items)
