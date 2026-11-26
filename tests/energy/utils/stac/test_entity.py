# tests/energy/utils/stac/test_entity.py
"""Tests for the energy extension's entity fields: entity types and IDs."""

import json

import pandas as pd
import pytest

from rbc.energy.utils.stac.entity import EntityTypeDefiner, entity_ids

EARLY = {"2020-01", "2020-02"}
LATE = {"2025-07"}


def _groups(rows: list[tuple], sysop: str = "ons") -> pd.DataFrame:
    """Run `EntityTypeDefiner.find_groups` on EGE rows, with the operator's vocabulary.

    Args:
        rows (list[tuple]): Per EGE: (id, name, code, lat, lon, months, total).
        sysop (str, optional): The operator. Defaults to "ons".

    Returns:
        pd.DataFrame: The parts found, with "entity_type" and "facility_name".
    """
    cols = ["id", "entity_name", "code", "lat", "lon", "months", "total"]
    df = pd.DataFrame(rows, columns=cols).set_index("id")
    return EntityTypeDefiner(sysop, {}).find_groups(
        df.drop(columns=["months", "total"]).reset_index(), df["months"], df["total"]
    )


def test_parts_by_code_take_their_shared_words() -> None:
    """Happy path: parts sharing a code need no points; unit words aren't in the name."""
    out = _groups(
        [
            ("a", "Santana Bloco 1", "CEG1", None, None, EARLY, 10.0),
            ("b", "Santana Bloco 2", "CEG1", None, None, EARLY, 20.0),
            ("c", "Other", "CEG2", None, None, EARLY, 5.0),
        ]
    )
    assert out.to_dict("index") == {
        "a": {"entity_type": "group", "facility_name": "Santana"},
        "b": {"entity_type": "group", "facility_name": "Santana"},
    }


def test_parts_by_code_without_shared_words_take_the_largest() -> None:
    """Happy path: without shared words, the facility is named after its largest part."""
    out = _groups(
        [
            ("a", "Belo Monte", "CEG1", None, None, EARLY, 900.0),
            ("b", "Pimental", "CEG1", None, None, EARLY, 50.0),
        ]
    )
    assert set(out["facility_name"]) == {"Belo Monte"}


def test_parts_by_name_must_lie_close() -> None:
    """Happy path: name-linked parts within 1 km group; farther ones or unlocated don't."""
    out = _groups(
        [
            ("a", "Angra I", "C1", -23.0, -44.46, EARLY, 1.0),
            ("b", "Angra II", "C2", -23.003, -44.46, EARLY, 1.0),  # ~0.3 km
            ("c", "Corumbá III", "C3", -16.0, -48.0, EARLY, 1.0),
            ("d", "Corumbá II", "C4", -16.5, -48.0, EARLY, 1.0),  # ~56 km
            ("e", "Paulo Afonso I", "C5", None, None, EARLY, 1.0),
            ("f", "Paulo Afonso II", "C6", None, None, EARLY, 1.0),  # no points
        ]
    )
    assert out.to_dict("index") == {
        "a": {"entity_type": "group", "facility_name": "Angra"},
        "b": {"entity_type": "group", "facility_name": "Angra"},
    }


def test_facility_name_spans_the_shared_discriminators() -> None:
    """Happy path: the name runs from the first to the last shared discriminator."""
    out = _groups(
        [
            ("a", "Porto do Pecém I", "C1", -3.5, -38.8, EARLY, 1.0),
            ("b", "Porto do Pecém II", "C2", -3.5, -38.8, EARLY, 2.0),
        ]
    )
    assert set(out["facility_name"]) == {"Porto do Pecém"}


def test_facility_name_keeps_shared_energy_words() -> None:
    """Happy path: an energy word all parts share stays in the name, a unit word not."""
    out = _groups(
        [
            ("a", "Fontes Solar I", "C1", None, None, EARLY, 1.0),
            ("b", "Fontes Solar II", "C1", None, None, EARLY, 2.0),
            ("c", "Lagoa Solar I", "C2", None, None, EARLY, 1.0),
            ("d", "Lagoa II", "C2", None, None, EARLY, 2.0),
        ]
    )
    assert out["facility_name"].to_dict() == {
        "a": "Fontes Solar",
        "b": "Fontes Solar",
        "c": "Lagoa",
        "d": "Lagoa",
    }


def test_code_style_names_are_split_at_underscores() -> None:
    """Happy path: a code-style name's words are separated by underscores, too."""
    out = _groups(
        [
            ("a", "TPP_NIKOLA_TESLA_B1", "W1", None, None, EARLY, 1.0),
            ("b", "TPP_NIKOLA_TESLA_B2", "W1", None, None, EARLY, 2.0),
        ]
    )
    assert set(out["facility_name"]) == {"TPP NIKOLA TESLA"}


def test_operator_with_vocabulary_per_country_is_refused() -> None:
    """Failure path: ENTSO-E's names need a vocabulary per country, not supported yet."""
    with pytest.raises(ValueError, match="per country"):
        _groups([("a", "Balti G09", "W1", None, None, EARLY, 1.0)], sysop="entsoe")


def test_successors_and_identical_names_are_no_parts() -> None:
    """Failure path: a renamed EGE (no common month) or one listed twice isn't a part."""
    out = _groups(
        [
            ("a", "Manauara", "CEG1", -3.1, -60.0, EARLY, 1.0),
            ("b", "Pirarucu", "CEG1", -3.1, -60.0, LATE, 1.0),
            ("c", "Itumbiara", "CEG2", -18.4, -49.1, EARLY, 1.0),
            ("d", "Itumbiara", "CEG2", -18.4, -49.1, EARLY, 0.0),
            ("e", "Santana Bloco 1", "CEG3", None, None, {"2018-01"}, 1.0),
            ("f", "Santana", "CEG3", None, None, {"2000-01", "2020-01"}, 1.0),
            ("g", "Termonorte I", "CEG4", None, None, EARLY, 0.0),
            ("h", "Termonorte 1", "CEG4", None, None, EARLY, 0.0),
        ]
    )
    assert out.empty
    assert list(out.columns) == ["entity_type", "facility_name"]


def test_parts_join_through_others() -> None:
    """Happy path: A-B by code and B-C by name make one facility of all three."""
    out = _groups(
        [
            ("a", "Mauá Bloco 3", "CEG1", None, None, EARLY, 5.0),
            ("b", "Mauá Bloco 4", "CEG1", -3.1, -60.0, EARLY, 1.0),
            ("c", "Mauá 3", "CEG2", -3.1, -60.001, EARLY, 9.0),
        ]
    )
    assert sorted(out.index) == ["a", "b", "c"]
    assert set(out["facility_name"]) == {"Mauá"}


def _entity_ids(rows: list[dict]) -> list[dict | None]:
    """Collect the entity IDs of EGEs given as coordinate finding's output rows.

    Args:
        rows (list[dict]): The EGEs' columns, "cen" for the operator's own ID.

    Returns:
        list[dict | None]: Each EGE's IDs, parsed from JSON.
    """
    df = pd.DataFrame(rows)
    own = {"cen": df.pop("cen").astype("string")}
    return [json.loads(ids) if ids else None for ids in entity_ids(df, own)]


def test_entity_ids_name_the_source_that_located_the_ege() -> None:
    """Happy path: the operator's own ID, plus the locating source's, fuzzy or not.

    osm-powerplants' IDs (ppdb) are OSM's own, so they share the "osm" key.
    """
    out = _entity_ids(
        [
            {"cen": "1", "match_method": "official_id_exact"},
            {"cen": "2", "match_method": "gem_name_fuzzy", "gem.id": "G1"},
            {"cen": None, "match_method": "ppdb_id_parent_fuzzy", "ppdb.id": "way/5"},
            {
                "cen": "4",
                "match_method": "osm_name_exact",
                "osm.id": 7.0,
                "osm.type": "node",
            },
        ]
    )
    assert out == [
        {"cen": "1"},
        {"cen": "2", "gem": "G1"},
        {"osm": "way/5"},
        {"cen": "4", "osm": "node/7"},
    ]


def test_entity_ids_skip_ids_that_name_another_ege() -> None:
    """Failure path: no ID from a match that wasn't kept, nor from a sibling's donor.

    A vetoed match keeps its locator columns in coordinate finding's output, a sibling
    inherits its donor's: both IDs belong to another EGE. PPM's 'id' is a row number.
    """
    out = _entity_ids(
        [
            {"cen": None, "match_method": "unmatched", "ppdb.id": "way/9"},
            {"cen": "2", "match_method": "gem_sibling", "gem.id": "G1"},
            {"cen": None, "match_method": "ppdb_id_exact", "ppdb.id": "123"},
        ]
    )
    assert out == [None, {"cen": "2"}, None]


def test_entity_ids_without_any_locators_columns() -> None:
    """Happy path: a run no locator matched in lacks their columns, which isn't an error."""
    assert _entity_ids([{"cen": "1", "match_method": "unmatched"}]) == [{"cen": "1"}]
