"""STAC RECORDS.

Describe a release as a STAC Collection with one Item per energy generating entity (EGE)..
"""

import json
import math
import re
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from loguru import logger
from pystac.errors import STACValidationError
from pystac.validation import validate_dict
from shapely.geometry import shape

from rbc.publish.icechunk_store import StorageInfo

JSON = dict[str, Any]  # a JSON object, e.g. a STAC Collection or Item

STAC_VERSION = "1.0.0"
VERSION_EXT = "https://stac-extensions.github.io/version/v1.2.0/schema.json"
ENERGY_EXT = (
    "https://energybench-association.github.io/stac-extension-energy/v0.1.0/schema.json"
)
THEMES_EXT = "https://stac-extensions.github.io/themes/v1.0.0/schema.json"
ITEM_ASSETS_EXT = "https://stac-extensions.github.io/item-assets/v1.0.0/schema.json"
DATACUBE_EXT = "https://stac-extensions.github.io/datacube/v2.2.0/schema.json"
SCIENTIFIC_EXT = "https://stac-extensions.github.io/scientific/v1.0.0/schema.json"
DOMAIN_SCHEME = "https://energybench.org/domains"
# Separator of a dataset's release Collections, e.g. "cen-pt1h__v0.1.0"
RELEASE_SEP = "__"

# The energy extension's Item properties, as columns of the STAC data (without "energy:")
ENERGY_FIELDS = [
    "entity_name",
    "entity_type",
    "facility_name",
    "group_name",
    "operator",
    "entity_ids",
    "fuel_type",
    "fuel_subtype",
    "hybrid_fuel_types",
    "generation_capacity",
    "storage_power",
    "status",
    "measurement_type",
]
# The fields summarized as value lists in the Collection (for filtering a search)
SUMMARY_FIELDS = ["fuel_type", "entity_type", "status", "measurement_type"]


@dataclass(frozen=True)
class DatasetInfo:
    """What a dataset's Collections say about its source, licence and processing.

    A dataset can come in several temporal resolutions, so its title, description and
    values name the resolution as `{tres}`, e.g. "hourly" or "15-minute", or as `{Tres}`
    at the start of a sentence (s. `for_resolution`).

    Attributes:
        dataset (str): ID of the dataset, e.g. "cen".
        title (str): Its title, e.g. "CEN (Chile) {tres} generation per plant".
        description (str): Its description, incl. licence terms and processing steps.
        license (str): The STAC licence value, e.g. "proprietary" for terms of use.
        license_url (str): Link to the licence's terms.
        providers (list[JSON]): STAC providers, e.g. the operator as "licensor".
        citation (str): How to cite the data (the scientific extension's citation).
        values (str): What the repository's `values` hold, e.g. "{Tres} gross
            generation (MW)".
        system_operator (str): The system operator's name, e.g. "Coordinador
            Eléctrico Nacional (CEN)".
        countries (dict[str, str]): The countries it covers: ISO 3166-1 alpha-2 code
            → name, e.g. {"CL": "Chile"}.
        continents (list[str]): The continents it covers, e.g. ["South America"].
        keywords (list[str]): Extra keywords, e.g. the country; the resolution is
            added to them.
    """

    dataset: str
    title: str
    description: str
    license: str
    license_url: str
    providers: list[JSON]
    citation: str
    values: str
    system_operator: str
    countries: dict[str, str]
    continents: list[str]
    keywords: list[str] = field(default_factory=list)

    def for_resolution(self, step: str) -> "DatasetInfo":
        """Fill the texts in for one temporal resolution.

        Args:
            step (str): The time step as an ISO 8601 duration, e.g. "PT1H", "PT15M".

        Returns:
            DatasetInfo: A copy whose texts name the resolution in words, e.g.
                "hourly" or "15-minute", and whose keywords end with it.

        Raises:
            ValueError: If the time step isn't a number of minutes, hours or days.
        """
        if match := re.fullmatch(r"PT(\d+)([HM])", step):
            count, unit = int(match[1]), {"H": "hour", "M": "minute"}[match[2]]
        elif match := re.fullmatch(r"P(\d+)D", step):
            count, unit = int(match[1]), "day"
        else:
            raise ValueError(f"'{step}' isn't a time step such as 'PT1H' or 'PT15M'.")
        if count == 1 and unit != "minute":
            word = "hourly" if unit == "hour" else "daily"
        else:
            word = f"{count}-{unit}"  # as in "15-minute generation"
        words = {"tres": word, "Tres": word.capitalize()}
        return replace(
            self,
            title=self.title.format(**words),
            description=self.description.format(**words),
            values=self.values.format(**words),
            keywords=[*self.keywords, word],
        )


def collection_id(dataset: str, step: str, release: str) -> str:
    """Give a dataset's Collection ID for one temporal resolution and release.

    Args:
        dataset (str): ID of the dataset, e.g. "cen".
        step (str): The time step as an ISO 8601 duration, e.g. "PT1H".
        release (str): The release, e.g. "v0.1.0".

    Returns:
        str: e.g. "cen-pt1h__v0.1.0".
    """
    return f"{dataset}-{step.lower()}{RELEASE_SEP}{release}"


# ------------------------------------------------------------------
# BUILDING
# ------------------------------------------------------------------
def build_items(
    df_stac: pd.DataFrame,
    spans: pd.DataFrame,
    info: DatasetInfo,
    release: str,
    storage: StorageInfo,
    snapshot: str,
    step: str,
) -> list[JSON]:
    """Build one Item per EGE that has any value in the release.

    An EGE without a single value has no time span to describe, so it gets no Item; it
    is logged by ID.

    Args:
        df_stac (pd.DataFrame): One row per EGE: "id", "geometry" (GeoJSON) and the
            energy fields (s. ENERGY_FIELDS) it has; empty values are left out.
        spans (pd.DataFrame): Each EGE's "start" and "end" (UTC), indexed by its ID.
        info (DatasetInfo): The dataset the Items belong to.
        release (str): The release, e.g. "v0.1.0".
        storage (StorageInfo): Where the release's repository lives.
        snapshot (str): The release tag's snapshot ID.
        step (str): The repository's time step as an ISO 8601 duration, e.g. "PT1H".

    Returns:
        list[JSON]: The Items, in the order of `df_stac`, linked as files of a static
            catalog (s. `write_stac`).

    Raises:
        ValueError: If an EGE has no geometry.
    """
    df = df_stac.join(spans, on="id")
    if (empty := df["start"].isna()).any():
        logger.warning(
            f"{int(empty.sum())} EGEs have no value at all, so no Item: "
            f"{df.loc[empty, 'id'].tolist()[:10]}"
        )
        df = df[~empty]
    if (no_geom := df["geometry"].isna()).any():
        raise ValueError(f"EGEs without geometry: {df.loc[no_geom, 'id'].tolist()[:5]}")

    cid = collection_id(info.dataset, step, release)
    info = info.for_resolution(step)
    created = _iso(pd.Timestamp.now(tz="UTC"))
    return [
        _item(row, cid, release, storage, snapshot, info, created)
        for _, row in df.iterrows()
    ]


def build_collection(
    info: DatasetInfo,
    release: str,
    items: list[JSON],
    storage: StorageInfo,
    snapshot: str,
    entities: list[str],
    times: pd.DatetimeIndex,
    step: str,
) -> JSON:
    """Build the release's Collection, summarizing its Items and describing its data.

    Args:
        info (DatasetInfo): The dataset.
        release (str): The release, e.g. "v0.1.0".
        items (list[JSON]): The release's Items.
        storage (StorageInfo): Where the release's repository lives.
        snapshot (str): The release tag's snapshot ID.
        entities (list[str]): The repository's `entity_id` values, in order.
        times (pd.DatetimeIndex): The repository's `time` axis (UTC, naive).
        step (str): The axis' time step as an ISO 8601 duration, e.g. "PT15M".

    Returns:
        JSON: The Collection, linked as the root of a static catalog (s. `write_stac`).
    """
    info = info.for_resolution(step)
    props = [it["properties"] for it in items]
    summaries: JSON = {}
    for f in SUMMARY_FIELDS:  # only those any Item has (STAC forbids empty lists)
        if values := sorted({p[f"energy:{f}"] for p in props if f"energy:{f}" in p}):
            summaries[f"energy:{f}"] = values
    capacities = [
        p["energy:generation_capacity"]
        for p in props
        if "energy:generation_capacity" in p
    ]
    if capacities:
        summaries["energy:generation_capacity"] = {
            "minimum": min(capacities),
            "maximum": max(capacities),
        }

    bboxes = [it["bbox"] for it in items]
    bbox = [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]
    interval = [
        min(p["start_datetime"] for p in props),
        max(p["end_datetime"] for p in props),
    ]
    cid = collection_id(info.dataset, step, release)
    return {
        "type": "Collection",
        "id": cid,
        "stac_version": STAC_VERSION,
        "stac_extensions": [
            VERSION_EXT,
            THEMES_EXT,
            ENERGY_EXT,
            ITEM_ASSETS_EXT,
            DATACUBE_EXT,
            SCIENTIFIC_EXT,
        ],
        "title": f"{info.title} — release {release}",
        "description": info.description,
        "license": info.license,
        "keywords": ["generation", release, *info.keywords],
        # the domain, then what the website's Browse filters offer (s. STAC contract)
        "themes": [
            {"scheme": DOMAIN_SCHEME, "concepts": [{"id": "generation"}]},
            {
                "scheme": "https://energybench.org/system-operators",
                "concepts": [{"id": info.dataset, "title": info.system_operator}],
            },
            {
                "scheme": "https://energybench.org/countries",  # ISO 3166-1 alpha-2
                "concepts": [
                    {"id": code, "title": name} for code, name in info.countries.items()
                ],
            },
            {
                "scheme": "https://energybench.org/continents",
                "concepts": [  # e.g. "south-america", "South America"
                    {"id": name.lower().replace(" ", "-"), "title": name}
                    for name in info.continents
                ],
            },
        ],
        "providers": info.providers,
        "version": release,
        "deprecated": False,
        "sci:citation": info.citation,
        "extent": {"spatial": {"bbox": [bbox]}, "temporal": {"interval": [interval]}},
        "summaries": summaries,
        "cube:dimensions": {
            "entity_id": {
                "type": "entity",
                "description": "Energy generating entity: its STAC Item's ID.",
                "values": entities,
            },
            "time": {
                "type": "temporal",
                "description": "Start of the time step (UTC).",
                "extent": [_iso(times[0]), _iso(times[-1])],
                "step": step,
            },
        },
        "cube:variables": {
            "values": {
                "type": "data",
                "dimensions": ["entity_id", "time"],
                "unit": "MW",
                "description": f"{info.values}; missing where not reported.",
            }
        },
        "item_assets": {
            "data": {
                "type": "application/vnd.icechunk",
                "roles": ["data"],
                "title": "Icechunk repository",
                "description": "The release's repository: the Item's entity's data is "
                "'values' at its 'icechunk:entity_id'.",
            }
        },
        "assets": {
            "data": {
                **_repository_asset(storage, release, snapshot),
                "description": f"All entities' data: 'values[entity_id, time]', "
                f"{info.values.lower()}.",
            }
        },
        "links": [
            {
                "rel": "license",
                "href": info.license_url,
                "type": "text/html",
                "title": "Terms of use",
            },
            {"rel": "root", "href": "./collection.json", "type": "application/json"},
            *(
                {
                    "rel": "item",
                    "href": f"./items/{it['id']}.json",
                    "type": "application/geo+json",
                }
                for it in items
            ),
        ],
    }


# ------------------------------------------------------------------
# CHECKING, WRITING, PUBLISHING
# ------------------------------------------------------------------
def validate(collection: JSON, items: list[JSON]) -> None:
    """Validate the Collection and every Item against STAC and each extension they use.

    Uses pystac's validation, which fetches the official schemas (once per run).

    Args:
        collection (JSON): The Collection.
        items (list[JSON]): Its Items.

    Raises:
        ValueError: If any of them is invalid, naming it and the reason.
    """
    for obj in [collection, *items]:
        try:
            validate_dict(obj)
        except STACValidationError as e:
            schema, reason, *_ = [*str(e).splitlines(), ""]
            raise ValueError(
                f"STAC {obj['type']} '{obj['id']}' is invalid: {reason} ({schema})"
            ) from e


def write_stac(out_dir: Path, collection: JSON, items: list[JSON]) -> Path:
    """Write the Collection and its Items as a static catalog (one JSON file each).

    Their relative links (root, item, collection, parent) point between these files.

    Args:
        out_dir (Path): The directory to write the Collection's folder into.
        collection (JSON): The Collection.
        items (list[JSON]): Its Items.

    Returns:
        Path: The Collection's folder, holding collection.json and items/<id>.json.
    """
    folder = Path(out_dir, collection["id"])
    Path(folder, "items").mkdir(parents=True, exist_ok=True)
    Path(folder, "collection.json").write_text(json.dumps(collection, indent=2))
    for item in items:
        Path(folder, "items", f"{item['id']}.json").write_text(json.dumps(item))
    logger.info(f"Wrote STAC Collection and {len(items)} Items to {folder}")
    return folder


def drop_stac(inbox: Path, collection: JSON, items: list[JSON]) -> Path:
    """Hand the Collection and its Items over through a folder the platform watches.

    They are written beside the folder, under `.incoming`, then moved in at once, so
    the platform never picks up a half-written release; one already waiting is replaced.

    Args:
        inbox (Path): The watched folder, e.g. on the LSDF.
        collection (JSON): The Collection.
        items (list[JSON]): Its Items.

    Returns:
        Path: The release's folder in the inbox.
    """
    staging = Path(inbox, ".incoming", collection["id"])
    shutil.rmtree(staging, ignore_errors=True)  # leftovers of an interrupted run
    write_stac(staging.parent, collection, items)
    target = Path(inbox, collection["id"])
    shutil.rmtree(target, ignore_errors=True)  # not yet picked up: replace it
    staging.rename(target)
    logger.info(f"Handed Collection '{collection['id']}' over in {target}")
    return target


def publish(api_url: str, collection: JSON, items: list[JSON]) -> None:
    """Create or replace the Collection and its Items in a STAC API (Transactions).

    The static catalog's relative links are left out: the API links its records itself.

    Args:
        api_url (str): The STAC API's root URL.
        collection (JSON): The Collection.
        items (list[JSON]): Its Items.

    Raises:
        requests.HTTPError: If the API refuses the Collection or an Item.
    """
    http = requests.Session()
    root = api_url.rstrip("/")
    targets = [(f"{root}/collections", collection)] + [
        (f"{root}/collections/{collection['id']}/items", item) for item in items
    ]
    for url, obj in targets:
        # only absolute links (e.g. the licence's): the API links its records itself
        body = {**obj, "links": [ln for ln in obj["links"] if "://" in ln["href"]]}
        r = http.post(url, json=body, timeout=60)
        if r.status_code == 409:  # it exists already: replace it
            r = http.put(f"{url}/{body['id']}", json=body, timeout=60)
        if not r.ok:
            raise requests.HTTPError(
                f"STAC API refused '{body['id']}' ({r.status_code}): {r.text[:300]}",
                response=r,
            )
    logger.info(
        f"Published Collection '{collection['id']}' with {len(items)} Items to {root}"
    )


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def _item(
    row: pd.Series,
    cid: str,
    release: str,
    storage: StorageInfo,
    snapshot: str,
    info: DatasetInfo,
    created: str,
) -> JSON:
    """Build one EGE's Item.

    Args:
        row (pd.Series): The EGE's STAC data, with its "start" and "end".
        cid (str): The Collection's ID.
        release (str): The release.
        storage (StorageInfo): Where the release's repository lives.
        snapshot (str): The release tag's snapshot ID.
        info (DatasetInfo): The dataset.
        created (str): When the Item was created (ISO 8601, UTC).

    Returns:
        JSON: The Item.
    """
    geometry = row["geometry"]
    props: JSON = {
        "datetime": None,
        "start_datetime": _iso(row["start"]),
        "end_datetime": _iso(row["end"]),
        "created": created,
        "title": row["entity_name"],
        "version": release,
    }
    for name in ENERGY_FIELDS:
        value = row.get(name)
        if name == "entity_ids" and isinstance(value, str):
            value = json.loads(value)
        # leave out what isn't given: None, NaN or an empty list/dict
        if value is None or value is pd.NA:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, (list, tuple, dict)) and not value:
            continue
        # as plain JSON values (not numpy/pandas scalars or tuples)
        if isinstance(value, tuple):
            value = list(value)
        elif hasattr(value, "item"):  # numpy scalar
            value = value.item()
        props[f"energy:{name}"] = value

    data: JSON = {
        **_repository_asset(storage, release, snapshot),
        "description": f"{info.values} of this entity: 'values' at entity_id "
        f"'{row['id']}', time in UTC.",
        "icechunk:entity_id": row["id"],
    }
    collection = {"href": "../collection.json", "type": "application/json"}
    return {
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": [VERSION_EXT, ENERGY_EXT],
        "id": row["id"],
        "collection": cid,
        "geometry": geometry,
        "bbox": list(shape(geometry).bounds),
        "properties": props,
        "assets": {"data": data},
        "links": [
            {"rel": "collection", **collection},
            {"rel": "parent", **collection},
            {"rel": "root", **collection},
        ],
    }


def _repository_asset(storage: StorageInfo, release: str, snapshot: str) -> JSON:
    """Describe the release's repository as a STAC asset.

    Args:
        storage (StorageInfo): Where the repository lives.
        release (str): The release (its tag).
        snapshot (str): The tag's snapshot ID.

    Returns:
        JSON: The asset's fields, without a description.
    """
    return {
        "href": storage.href,  # the permanent data URL, never the storage itself
        "type": "application/vnd.icechunk",
        "title": f"Icechunk repository @ {release}",
        "roles": ["data"],
        "icechunk:repository": storage.prefix,
        "icechunk:tag": release,
        "icechunk:snapshot": snapshot,
    }


def _iso(ts: pd.Timestamp) -> str:
    """Write a UTC timestamp as STAC wants it, e.g. '2024-01-01T03:00:00Z'.

    Args:
        ts (pd.Timestamp): The timestamp (tz-aware, or naive UTC).

    Returns:
        str: The ISO 8601 text with a 'Z'.
    """
    utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")
