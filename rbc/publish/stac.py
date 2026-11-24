"""STAC RECORDS.

Describe a release as STAC records: a Collection's dataset-level fields, which any kind of
data has (s. `build_collection_base`), the energy datasets' Collection with one Item per
energy generating entity (EGE, s. `EnergyStacBuilder`), and checking, writing and
publishing them.
"""

import json
import math
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field, replace
from functools import cache
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

# STAC-related global variables (incl. extension schemas)
STAC_VERSION = "1.0.0"
VERSION_EXT = "https://stac-extensions.github.io/version/v1.2.0/schema.json"
ENERGY_EXT = (
    "https://energybench-association.github.io/stac-extension-energy/v0.1.0/schema.json"
)
THEMES_EXT = "https://stac-extensions.github.io/themes/v1.0.0/schema.json"
ITEM_ASSETS_EXT = "https://stac-extensions.github.io/item-assets/v1.0.0/schema.json"
DATACUBE_EXT = "https://stac-extensions.github.io/datacube/v2.2.0/schema.json"
SCIENTIFIC_EXT = "https://stac-extensions.github.io/scientific/v1.0.0/schema.json"
PROCESSING_EXT = "https://stac-extensions.github.io/processing/v1.2.0/schema.json"

# The project's metadata: name and version of the software that processes the data
PYPROJECT = Path(Path(__file__).resolve().parents[2], "pyproject.toml")

# The energy extension's Item properties (= df_stac columns), written by `EnergyStacBuilder`
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
    """What a dataset's Collections say about its source, license and processing.

    A dataset can come in several temporal resolutions, so its title, description,
    lineage and values name the resolution as `{tres}`, e.g. "hourly" or "15-minute", or
    as `{Tres}` at the start of a sentence (s. `for_resolution`).

    Attributes:
        dataset (str): ID of the dataset, e.g. "cen".
        title (str): Its title, e.g. "CEN (Chile) {tres} generation per plant".
        description (str): Its description, incl. license terms.
        license (str): The STAC license value, e.g. "proprietary" for terms of use.
        license_url (str): Link to the license's terms.
        providers (list[JSON]): STAC providers, e.g. the operator as "licensor" and the
            "EnergyBench Association" as "processor".
        citation (str): How to cite the data (the scientific extension's citation).
        lineage (str): How the source's data was changed: the processing extension's
            lineage, on the "processor" provider, also closing the description.
        values (str): What the repository's `values` hold, e.g. "{Tres} gross
            generation (MW)".
        system_operator (str): The system operator's name, e.g. "Coordinador
            Eléctrico Nacional (CEN)".
        countries (dict[str, str]): The countries it covers: ISO 3166-1 alpha-2 code → name,
            e.g. {"CL": "Chile"}.
        continents (list[str]): The continents it covers, e.g. ["South America"].
        keywords (list[str]): Extra keywords, e.g. the country; the resolution is
            added to them.
        license_notice_url (str | None): Link to where the source states its license,
            if not at `license_url`. Defaults to None.
    """

    dataset: str
    title: str
    description: str
    license: str
    license_url: str
    providers: list[JSON]
    citation: str
    lineage: str
    values: str
    system_operator: str
    countries: dict[str, str]
    continents: list[str]
    keywords: list[str] = field(default_factory=list)
    license_notice_url: str | None = None

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
            lineage=self.lineage.format(**words),
            values=self.values.format(**words),
            keywords=[*self.keywords, word],
        )


@cache
def get_software() -> dict[str, str]:
    """Define the STAC processing:software as project name and version plus git commit.

    Read from the checked-out pyproject.toml, so both match the code that runs (an
    editable install's metadata keeps the version it was installed with). The version
    alone isn't raised per change, so the commit tells which code produced a release,
    e.g. "0.0.1+g4176167" (a PEP 440 local version).

    Returns:
        dict[str, str]: The processing extension's software, e.g. {"RenewBench-Crawler":
            "0.0.1+g4176167"}. Fallback: import package's name and "unknown" without
            pyproject.toml; version without "+g<commit>" outside a git checkout.
    """
    try:
        project = tomllib.loads(PYPROJECT.read_text())["project"]
        name, package = project["name"], project["version"]

    except (OSError, KeyError, tomllib.TOMLDecodeError):
        name, package = __name__.split(".")[0], "unknown"

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {name: package}  # no git checkout or no git exists

    return {name: f"{package}+g{commit}" if commit else package}


# ------------------------------------------------------------------
# BUILDING (general)
# ------------------------------------------------------------------
def build_collection_base(info: DatasetInfo, release: str, cid: str) -> JSON:
    """Build a Collection's dataset-level fields, the same for any kind of data.

    What the dataset says about itself: title, description (incl. lineage), license and
    license links, providers (processing extension), version, citation.
    What the data looks like (extent, summaries, datacube, assets, Item links)
    is added by a builder for the specific kind of data (e.g. `EnergyStacBuilder` for EG).

    Args:
        info (DatasetInfo): The dataset, its texts filled in (s. `for_resolution`).
        release (str): The release, e.g. "v0.1.0".
        cid (str): The Collection's ID (s. `_collection_id`).

    Returns:
        JSON: The Collection's fields, with the extensions they use and the license
            links.

    Raises:
        ValueError: If no provider is a "processor", to hold the lineage.
    """
    processing = {
        "processing:lineage": info.lineage,  # for "processor" provider (= EnergyBench)
        "processing:software": get_software(),
    }
    providers = [
        {**p, **processing} if "processor" in p.get("roles", []) else p
        for p in info.providers
    ]
    if providers == info.providers:
        raise ValueError(
            f"'{info.dataset}' has no 'processor' provider for its lineage."
        )

    license_links = [
        {
            "rel": "license",
            "href": info.license_url,
            "type": "text/html",
            "title": "License",
        }
    ]
    if info.license_notice_url:
        license_links.append(
            {
                "rel": "license",
                "href": info.license_notice_url,
                "type": "text/html",
                "title": "License notice of the source",
            }
        )
    return {
        "type": "Collection",
        "id": cid,
        "stac_version": STAC_VERSION,
        "stac_extensions": [VERSION_EXT, SCIENTIFIC_EXT, PROCESSING_EXT],
        "title": f"{info.title} — release {release}",
        "description": f"{info.description.rstrip()}\n\n{info.lineage}",
        "license": info.license,
        "providers": providers,
        "version": release,
        "deprecated": False,
        "sci:citation": info.citation,
        "links": license_links,
    }


# ------------------------------------------------------------------
# BUILDING (energy)
# ------------------------------------------------------------------
class EnergyStacBuilder:
    """Builds the STAC records of one release of an energy dataset, in one resolution.

    One Item per EGE with the energy extension's fields, and one Collection summarizing
    them and describing the release's Icechunk repository ('values[entity_id, time]' in
    MW). The Collection's dataset-level fields come from `build_collection_base`.

    Attributes:
        dataset (DatasetInfo): The dataset, its texts filled in for the resolution.
        release (str): The release, e.g. "v0.1.0".
        storage (StorageInfo): Where the release's repository lives.
        snapshot (str): The release tag's snapshot ID.
        step (str): The repository's time step as an ISO 8601 duration, e.g. "PT1H".
        cid (str): The Collection's ID, e.g. "cen-pt1h__v0.1.0".
    """

    def __init__(
        self,
        dataset: DatasetInfo,
        release: str,
        storage: StorageInfo,
        snapshot: str,
        step: str,
    ) -> None:
        """Initialize the builder.

        Args:
            dataset (DatasetInfo): The dataset (s. `DatasetInfo.for_resolution`).
            release (str): The release, e.g. "v0.1.0".
            storage (StorageInfo): Where the release's repository lives.
            snapshot (str): The release tag's snapshot ID.
            step (str): The repository's time step as an ISO 8601 duration, e.g. "PT1H".
        """
        self.dataset = dataset.for_resolution(step)
        self.release = release
        self.storage = storage
        self.snapshot = snapshot
        self.step = step
        self.cid = _collection_id(dataset.dataset, step, release)

    def build_items(self, df_stac: pd.DataFrame, spans: pd.DataFrame) -> list[JSON]:
        """Build one Item per EGE that has any value in the release.

        An EGE without a single value has no time span to describe, so it gets no Item;
        it is logged by ID.

        Args:
            df_stac (pd.DataFrame): One row per EGE: "id", "geometry" (GeoJSON) and the
                energy fields (s. ENERGY_FIELDS) it has; empty values are left out.
            spans (pd.DataFrame): Each EGE's "start" and "end" (UTC), indexed by its ID.

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
            raise ValueError(
                f"EGEs without geometry: {df.loc[no_geom, 'id'].tolist()[:5]}"
            )

        created = _utc_to_iso(pd.Timestamp.now(tz="UTC"))
        return [self._build_item(row, created) for _, row in df.iterrows()]

    def build_collection(
        self, items: list[JSON], entities: list[str], times: pd.DatetimeIndex
    ) -> JSON:
        """Build the release's Collection, summarizing its Items and describing its data.

        Args:
            items (list[JSON]): The release's Items (s. `build_items`).
            entities (list[str]): The repository's `entity_id` values, in order.
            times (pd.DatetimeIndex): The repository's `time` axis (UTC, naive).

        Returns:
            JSON: The Collection, linked as the root of a static catalog (s.
                `write_stac`).

        Raises:
            ValueError: If no provider is a "processor", to hold the lineage.
        """
        base = build_collection_base(self.dataset, self.release, self.cid)

        props = [it["properties"] for it in items]
        summaries: JSON = {}
        for f in SUMMARY_FIELDS:  # only those any Item has (STAC forbids empty lists)
            if values := sorted(
                {p[f"energy:{f}"] for p in props if f"energy:{f}" in p}
            ):
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
        return {
            **base,
            "stac_extensions": [
                *base["stac_extensions"],
                THEMES_EXT,
                ENERGY_EXT,
                ITEM_ASSETS_EXT,
                DATACUBE_EXT,
            ],
            "keywords": ["generation", self.release, *self.dataset.keywords],
            # the domain, then what the website's Browse filters offer (s. STAC contract)
            "themes": [
                {
                    "scheme": "https://energybench.org/domains",
                    "concepts": [{"id": "generation"}],
                },
                {
                    "scheme": "https://energybench.org/system-operators",
                    "concepts": [
                        {
                            "id": self.dataset.dataset,
                            "title": self.dataset.system_operator,
                        }
                    ],
                },
                {
                    "scheme": "https://energybench.org/countries",  # ISO 3166-1 alpha-2
                    "concepts": [
                        {"id": code, "title": name}
                        for code, name in self.dataset.countries.items()
                    ],
                },
                {
                    "scheme": "https://energybench.org/continents",
                    "concepts": [  # e.g. "south-america", "South America"
                        {"id": name.lower().replace(" ", "-"), "title": name}
                        for name in self.dataset.continents
                    ],
                },
            ],
            "extent": {
                "spatial": {"bbox": [bbox]},
                "temporal": {"interval": [interval]},
            },
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
                    "extent": [_utc_to_iso(times[0]), _utc_to_iso(times[-1])],
                    "step": self.step,
                },
            },
            "cube:variables": {
                "values": {
                    "type": "data",
                    "dimensions": ["entity_id", "time"],
                    "unit": "MW",
                    "description": f"{self.dataset.values}; missing where not reported.",
                }
            },
            "item_assets": {
                "data": {
                    "type": "application/vnd.icechunk",
                    "roles": ["data"],
                    "title": "Icechunk repository",
                    "description": "The release's repository: the Item's entity's data "
                    "is 'values' at its 'icechunk:entity_id'.",
                }
            },
            "assets": {
                "data": {
                    **_icechunk_repo_as_asset(
                        self.storage, self.release, self.snapshot
                    ),
                    "description": f"All entities' data: 'values[entity_id, time]', "
                    f"{self.dataset.values.lower()}.",
                }
            },
            "links": [
                *base["links"],
                {
                    "rel": "root",
                    "href": "./collection.json",
                    "type": "application/json",
                },
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

    def _build_item(self, row: pd.Series, created: str) -> JSON:
        """Build one EGE's Item.

        Args:
            row (pd.Series): The EGE's STAC data, with its "start" and "end".
            created (str): When the Item was created (ISO 8601, UTC).

        Returns:
            JSON: The Item.
        """
        geometry = row["geometry"]
        props: JSON = {
            "datetime": None,
            "start_datetime": _utc_to_iso(row["start"]),
            "end_datetime": _utc_to_iso(row["end"]),
            "created": created,
            "title": row["entity_name"],
            "version": self.release,
        }
        for name in ENERGY_FIELDS:
            value = row.get(name)
            if name == "entity_ids" and isinstance(value, str):
                value = json.loads(value)

            # leave out what's not given (None, NaN, empty list/dict)
            if value is None or value is pd.NA:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            if isinstance(value, (list, tuple, dict)) and not value:
                continue

            # define as plain JSON values (not np/pd scalars or tuples)
            if isinstance(value, tuple):
                value = list(value)
            elif hasattr(value, "item"):  # np scalar
                value = value.item()

            props[f"energy:{name}"] = value

        data: JSON = {
            **_icechunk_repo_as_asset(self.storage, self.release, self.snapshot),
            "description": f"{self.dataset.values} of this entity: 'values' at entity_id "
            f"'{row['id']}', time in UTC.",
            "icechunk:entity_id": row["id"],
        }
        collection = {"href": "../collection.json", "type": "application/json"}
        return {
            "type": "Feature",
            "stac_version": STAC_VERSION,
            "stac_extensions": [VERSION_EXT, ENERGY_EXT],
            "id": row["id"],
            "collection": self.cid,
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


def publish_stac(api_url: str, collection: JSON, items: list[JSON]) -> None:
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
        # only absolute links (e.g. the license's): the API links its records itself
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
def _collection_id(dataset: str, step: str, release: str) -> str:
    """Specify a dataset's Collection ID based on the temporal resolution and release.

    Args:
        dataset (str): ID of the dataset, e.g. "cen".
        step (str): The time step as an ISO 8601 duration, e.g. "PT1H".
        release (str): The release, e.g. "v0.1.0".

    Returns:
        str: The release collection ID, e.g. "cen-pt1h__v0.1.0".
    """
    return f"{dataset}-{step.lower()}__{release}"


def _icechunk_repo_as_asset(storage: StorageInfo, release: str, snapshot: str) -> JSON:
    """Describe the release's Icechunk repository as a STAC asset.

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


def _utc_to_iso(ts: pd.Timestamp) -> str:
    """Write a UTC timestamp as STAC wants it, e.g. '2024-01-01T03:00:00Z'.

    Args:
        ts (pd.Timestamp): The timestamp (tz-aware, or naive UTC).

    Returns:
        str: The ISO 8601 text with a 'Z'.
    """
    utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")
