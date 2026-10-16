"""CEN API REGISTRY COORDINATE PROCESSOR."""

import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from pyproj import Transformer

from rbc.energy.cen.downloader import URL_ROOT
from rbc.energy.utils import MAX_RETRIES, DataStructureError, load_df_from_file

# API registry of CEN's EGEs (returns all EGEs incl. retired regardless of requested date)
URL_REGISTRY = f"{URL_ROOT}centrales/v4/findByDate"

# Chilean territory bounding box (incl. islands) to drop any broken coordinates
CHILE_LAT = (-56.5, -17.0)
CHILE_LON = (-110.0, -66.0)


class CenCoordinateRegistry:
    """CEN EGE registry coordinate processor.

    Downloads and converts the CEN data from the `centrales` endpoint to build the CEN
    coordinate registry (unique EGE infos and their coordinates). The registry also
    contains retired EGEs, so it covers older generation data as well.

    Attributes:
        token (str): The CEN API token.
        cache_dir (Path): Path to the cache directory.
        df_registry (pd.DataFrame): Plant registry with lat/lon, one row per EGE ID.
    """

    def __init__(self, token: str, cache_dir: Path):
        """Initialize the CEN EGE registry processor.

        Loads the registry from cache ('centrales_<year>.csv' in cache_dir) if present,
        otherwise downloads it fresh. The cache holds the API's fields as delivered, so
        lat/lon are converted from its UTM fields on every load.

        Args:
            token (str): The CEN API token.
            cache_dir (Path): Path to the cache directory for the CEN API-parsed coord data.
        """
        self.token = token
        self.cache_dir = Path(cache_dir)

        self.year = date.today().year
        cache_path = Path(self.cache_dir, f"centrales_{self.year}.csv")
        if cache_path.is_file():
            logger.info(f"Loading CEN EGE registry from cache: {cache_path}")
            # UTM fields as text: their separators decide the value ('300.000' is 300 km)
            utm_as_text = {
                c: str for c in ("coordenada_este", "coordenada_norte", "zona_huso")
            }
            df = load_df_from_file(cache_path, dtype=utm_as_text)
        else:
            df = self.fetch_registry()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
            logger.info(f"Cached CEN EGE registry to: {cache_path}")

        self.df_registry = add_latlon(df)

    def fetch_registry(self) -> pd.DataFrame:
        """Download the CEN EGE registry, keeping the API's fields as delivered.

        Returns:
            pd.DataFrame: One row per EGE with the API's own field names: ['id_central',
                'central', 'tipo_tecnologia', 'estado', 'comuna', 'region',
                'coordenada_este', 'coordenada_norte', 'zona_huso'].

        Raises:
            requests.RequestException: If a request still fails after all retries.
            DataStructureError: If the response structure changed.
        """
        day = f"{self.year}-01-01"  # required, but has no effect on the result
        params: dict[str, str | int] = {
            "startDate": day,
            "endDate": day,
            "user_key": self.token,
            "limit": 1500,  # as of 2026, 1362 EGEs are parsed and this works in one go
            "page": 1,
        }

        rows: list[dict] = []
        attempt = 0
        while True:
            try:
                response = requests.get(URL_REGISTRY, params=params, timeout=120)
                response.raise_for_status()
            except requests.RequestException as e:
                if attempt >= MAX_RETRIES:
                    raise
                attempt += 1
                logger.warning(f"Registry request failed ({type(e).__name__}), retry")
                time.sleep(5)
                continue

            try:
                body = response.json()
                rows.extend(body["data"])
                if int(params["page"]) >= body["totalPages"]:
                    break
            except (ValueError, KeyError) as e:
                raise DataStructureError(
                    f"CEN registry structure changed: {type(e).__name__} {e}"
                ) from e
            params["page"] = int(params["page"]) + 1

        try:
            df = pd.DataFrame(
                {
                    "id_central": [int(r["id_central"]) for r in rows],
                    "central": [str(r["central"]) for r in rows],
                    "tipo_tecnologia": [r.get("tipo_tecnologia") for r in rows],
                    "estado": [r.get("estado") or None for r in rows],
                    "comuna": [r.get("comuna") for r in rows],
                    "region": [r.get("region") for r in rows],
                    "coordenada_este": [r.get("coordenada_este") for r in rows],
                    "coordenada_norte": [r.get("coordenada_norte") for r in rows],
                    "zona_huso": [r.get("zona_huso") for r in rows],
                }
            )
        except KeyError as e:
            raise DataStructureError(f"CEN registry is missing column {e}") from e

        return df.drop_duplicates("id_central").reset_index(drop=True)


def add_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """Add lat/lon (WGS84) columns, converted from the registry's UTM fields.

    Args:
        df (pd.DataFrame): Registry rows with the API's UTM fields, as text.

    Returns:
        pd.DataFrame: The rows with 'lat' and 'lon' added (empty where not convertible).
    """
    latlon = [
        utm_to_latlon(east, north, zone)
        for east, north, zone in zip(
            df["coordenada_este"], df["coordenada_norte"], df["zona_huso"]
        )
    ]
    df = df.assign(lat=[lat for lat, _ in latlon], lon=[lon for _, lon in latlon])

    logger.info(
        f"Loaded {len(df)} EGEs from CEN coordinate registry, "
        f"{int(df['lat'].notna().sum())} with valid coordinates."
    )
    return df


def utm_to_latlon(easting, northing, zone) -> tuple[float | None, float | None]:
    """Convert UTM coordinates as reported by CEN to lat/lon (WGS84).

    Uses the EGEs's own UTM zone (Chilean EGEs lie in zones 18 and 19) and handles
    CEN's number formats, e.g. '679310,49', '6.287.533,3', '300.000', '5762380.85'.

    Args:
        easting (str | None): Easting in metres ('coordenada_este').
        northing (str | None): Northing in metres ('coordenada_norte').
        zone (str | None): UTM zone, e.g. '18G', '19 H' ('zona_huso').

    Returns:
        tuple[float | None, float | None]: (lat, lon), or (None, None) if values are
            missing, unparsable or the point lies outside Chile.
    """
    try:
        numbers = []
        for text in (easting, northing):
            t = str(text).replace(" ", "")

            # '.' = thousands separator if a decimal comma is present ('6.287.533,3')
            # or if it groups digits in threes ('300.000'), else decimal ('5762380.85')
            if "," in t or re.fullmatch(r"\d{1,3}(\.\d{3})+", t):
                t = t.replace(".", "")
            numbers.append(float(t.replace(",", ".")))
    except ValueError:
        return None, None

    zone_match = re.search(r"\d{1,2}", str(zone))
    if zone_match is None or not 1 <= (zone_number := int(zone_match.group())) <= 60:
        return None, None

    utm_crs = f"EPSG:327{zone_number:02d}"  # UTM zones in the southern hemisphere
    lon, lat = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True).transform(
        numbers[0], numbers[1]
    )
    if CHILE_LAT[0] <= lat <= CHILE_LAT[1] and CHILE_LON[0] <= lon <= CHILE_LON[1]:
        return float(lat), float(lon)
    return None, None


if __name__ == "__main__":
    reg = CenCoordinateRegistry(
        token="<cen_token>", cache_dir=Path("my/path/cen/coordinates")
    )
