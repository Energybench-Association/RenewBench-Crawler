"""DATA SOURCE CONFIGURATION.

Schema definitions for different data sources.
"""

from pathlib import Path
from typing import Literal, Type

from pydantic import BaseModel, field_validator, model_validator


# ----------------------------------
# General schemas
# ----------------------------------
class Paths(BaseModel):
    """Filesystem paths used by a data source.

    Attributes:
        dst_dir_raw (Path): Destination directory for raw data output.
        dst_dir_proc (Path): Destination directory for processed data outputs.
    """

    dst_dir_raw: Path
    dst_dir_proc: Path | None = None

    @model_validator(mode="after")
    def _default_proc_dir(self) -> "Paths":
        """Define a default dst_dir_proc path if none was provided."""
        if self.dst_dir_proc is None:
            dst_dir_proc = Path(str(self.dst_dir_raw).replace("/raw/", "/processed/"))

            if dst_dir_proc == self.dst_dir_raw:
                sysop = self.dst_dir_raw.parts[-1]
                dst_dir_proc = Path(self.dst_dir_raw.parent, "processed", sysop)

            self.dst_dir_proc = dst_dir_proc

        return self


class AccessAPI(BaseModel):
    """Access settings for a data source requiring API key / security token.

    Attributes:
        api_key (str): API key.
    """

    api_key: str


class AccessAccount(BaseModel):
    """Access settings for a data source requiring an account username & password.

    Attributes:
        username (str): Account username.
        password (str): Account password.
    """

    username: str
    password: str


# ----------------------------------
# General schema validators
# ----------------------------------
class PathValidation:
    """Validator for Path fields to ensure they are safe and writable."""

    @field_validator("paths", mode="after")
    @classmethod
    def check_paths(cls, paths: BaseModel) -> BaseModel:
        """Validate that destination directories are accessible.

        Args:
            paths (BaseModel): Parsed paths configuration.

        Returns:
            BaseModel: The validated paths object.

        Raises:
            ValueError: If the path is not writable.
        """
        for field, path in paths.model_dump().items():
            if not isinstance(path, (str, Path)) or field == "dst_dir_proc":
                continue  # created by the processing writing into it, not by every script

            try:
                Path(path).mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError) as e:
                raise ValueError(
                    f"Path '{field}' ('{path}') is not writable or cannot be created. "
                    f"System error: {e}"
                )

        return paths


class AccessValidation:
    """Validator for "access" fields to ensure strings contain non-placeholder content."""

    @field_validator("access", mode="after")
    @classmethod
    def check_access_values(cls, access: BaseModel) -> BaseModel | None:
        """Validate that access credentials contain real values.

        Args:
            access (BaseModel): Parsed access configuration.

        Returns:
            BaseModel | None: The validated access object.

        Raises:
            ValueError: If an access field is empty or contains a placeholder.
        """
        if access is None:
            return access

        for field, value in access.model_dump().items():
            if not isinstance(value, str):
                continue
            if not value.strip():
                raise ValueError(f"Access field '{field}' is empty")
            if any(marker in value for marker in ("YOUR-SECRET", "COMMIT")):
                raise ValueError(
                    f"Access field '{field}' still contains the placeholder '{value}'!"
                )
        return access


# ----------------------------------
# Per-source schemas - ENERGY
# ----------------------------------
class AdmeConfig(PathValidation, BaseModel):
    """Configuration schema for the ADME energy data source.

    Attributes:
        source (Literal): Name of the data source.
        paths (Paths): Paths pydantic model for paths.
    """

    source: Literal["adme"] = "adme"
    paths: Paths


class AemoConfig(PathValidation, AccessValidation, BaseModel):
    """Configuration schema for the AEMO energy data source.

    Attributes:
        source (Literal): Name of the data source.
        paths (Paths): Paths pydantic model for paths.
        access (AccessAPI): Access pydantic model for access settings.
    """

    source: Literal["aemo"] = "aemo"
    paths: Paths
    access: AccessAPI


class AesoConfig(PathValidation, AccessValidation, BaseModel):
    """Configuration schema for the AESO energy data source.

    Attributes:
        source (Literal): Name of the data source.
        paths (Paths): Paths pydantic model for paths.
        access (AccessAPI): Access pydantic model for access settings.
    """

    source: Literal["aeso"] = "aeso"
    paths: Paths
    access: AccessAPI


class CenConfig(PathValidation, AccessValidation, BaseModel):
    """Configuration schema for the CEN energy data source.

    Attributes:
        source (Literal): Name of the data source.
        paths (Paths): Paths pydantic model for paths.
        access (AccessAPI): Access pydantic model for access settings.
    """

    source: Literal["cen"] = "cen"
    paths: Paths
    access: AccessAPI


class EatConfig(PathValidation, BaseModel):
    """Configuration schema for the EAT energy data source.

    Attributes:
        source (Literal): Name of the data source.
        paths (Paths): Paths pydantic model for paths.
    """

    source: Literal["eat"] = "eat"
    paths: Paths


class EiaConfig(PathValidation, AccessValidation, BaseModel):
    """Configuration schema for the EIA energy data source.

    Attributes:
        source (Literal): Name of the data source.
        paths (Paths): Paths pydantic model for paths.
        access (AccessAPI): Access pydantic model for access settings.
    """

    source: Literal["eia"] = "eia"
    paths: Paths
    access: AccessAPI


class EntsoeConfig(PathValidation, AccessValidation, BaseModel):
    """Configuration schema for the ENTSO-E energy data source.

    Attributes:
        source (Literal): Name of the data source.
        paths (Paths): Paths pydantic model for paths.
        access (AccessAPI): Access pydantic model for access settings.
    """

    source: Literal["entsoe"] = "entsoe"
    paths: Paths
    access: AccessAPI


class EpiasConfig(PathValidation, AccessValidation, BaseModel):
    """Configuration schema for the EPIAS energy data source.

    Attributes:
        source (Literal): Name of the data source.
        paths (Paths): Paths pydantic model for paths.
        access (AccessAccount): Access pydantic model for access settings.
    """

    source: Literal["epias"] = "epias"
    paths: Paths
    access: AccessAccount


class IesoConfig(PathValidation, BaseModel):
    """Configuration schema for the IESO energy data source.

    Attributes:
        source (Literal): Name of the data source.
        paths (Paths): Paths pydantic model for paths.
    """

    source: Literal["ieso"] = "ieso"
    paths: Paths


class OnsConfig(PathValidation, BaseModel):
    """Configuration schema for the ONS energy data source.

    Attributes:
        source (Literal): Name of the data source.
        paths (Paths): Paths pydantic model for paths.
    """

    source: Literal["ons"] = "ons"
    paths: Paths


class ReiConfig(PathValidation, BaseModel):
    """Configuration schema for the REI energy data source.

    Attributes:
        source (Literal): Name of the data source.
        paths (Paths): Paths pydantic model for paths.
    """

    source: Literal["rei"] = "rei"
    paths: Paths


# ----------------------------------
# Per-source schemas - WEATHER
# ----------------------------------
class Era5Config(PathValidation, AccessValidation, BaseModel):
    """Configuration schema for the ERA5 reanalysis data source.

    Attributes:
        source (Literal): Name of the data source.
        paths (Paths): Paths pydantic model for paths.
        access (AccessAPI): Access pydantic model for access settings.
    """

    source: Literal["era5"] = "era5"
    paths: Paths
    access: AccessAPI


class IconDreamGlobalConfig(PathValidation, BaseModel):
    """Configuration schema for the ICON-DREAM Global weather data source.

    Attributes:
        source (Literal): Name of the data source.
        paths (Paths): Paths pydantic model for paths.
    """

    source: Literal["icon_dream_global"] = "icon_dream_global"
    paths: Paths


class IconDreamEuConfig(PathValidation, BaseModel):
    """Configuration schema for the ICON-DREAM EU weather data source.

    Attributes:
        source (Literal): Name of the data source.
        paths (Paths): Paths pydantic model for paths.
    """

    source: Literal["icon_dream_eu"] = "icon_dream_eu"
    paths: Paths


class Barra2Config(PathValidation, BaseModel):
    """Configuration schema for the BARRA2 reanalysis weather data source.

    Attributes:
        source (Literal): Name of the data source.
        paths (Paths): Paths pydantic model for paths.
    """

    source: Literal["barra2"] = "barra2"
    paths: Paths


# ----------------------------------
# Coordinate finding schemas
# ----------------------------------
class CoordinatesPaths(BaseModel):
    """Filesystem paths used by coordinate finding (shared by all energy sources).

    Attributes:
        resources_dir (Path): Directory of the resources coordinate finding downloads or
            reads (e.g. GEM trackers in `gem/`, the EIC directory, OSM files).
    """

    resources_dir: Path


class CoordinatesConfig(PathValidation, BaseModel):
    """Configuration schema for coordinate finding.

    Attributes:
        source (Literal): Name of the config.
        paths (CoordinatesPaths): Paths pydantic model for paths.
    """

    source: Literal["coordinates"] = "coordinates"
    paths: CoordinatesPaths


# ----------------------------------
# Publishing schemas
# ----------------------------------
class PublishLsdf(BaseModel):
    """The LSDF route: write to the mounted LSDF, the VM publishes from there.

    Attributes:
        dst_dir_data (Path): Directory holding the bucket's directory.
        dst_dir_inbox (Path): Folder the VM picks the STAC records up from.
    """

    dst_dir_data: Path
    dst_dir_inbox: Path


class PublishS3(BaseModel):
    """The S3 route: write over S3, publish to the STAC API directly.

    Attributes:
        url_endpoint (str): S3 endpoint the repositories are written to.
        url_stac_api (str): Root URL of the STAC API (with Transactions).
    """

    url_endpoint: str
    url_stac_api: str


class PublishChunks(BaseModel):
    """Chunk shape of the generation arrays.

    Attributes:
        entities (int): EGEs per chunk.
        days (int): Days of data per chunk, whatever the data's temporal resolution
            (e.g. 365 days = 8760 hourly or 35040 15-minute steps).
    """

    entities: int = 16
    days: int = 365


class PublishConfig(BaseModel):
    """Configuration schema for publishing processed data to EnergyBench.

    Attributes:
        source (Literal): Name of the config.
        route (Literal["lsdf", "s3"]): How the data is published (s. `lsdf`, `s3`).
        release (str): The release to write and publish, e.g. "v0.1.0".
        bucket (str): Bucket of the repositories, on either route.
        url_public (str): Permanent public URL the STAC assets name, on either route
            (only used when publishing: else they name the local repository).
        lsdf (PublishLsdf): Settings of the LSDF route.
        s3 (PublishS3): Settings of the S3 route.
        chunks (PublishChunks): Chunk shape of the generation arrays.
    """

    source: Literal["publish"] = "publish"
    route: Literal["lsdf", "s3"]
    release: str
    bucket: str = "energybench"
    url_public: str
    lsdf: PublishLsdf
    s3: PublishS3
    chunks: PublishChunks = PublishChunks()

    def check_publishable(self) -> None:
        """Check the URLs publishing on the chosen route needs hold real values.

        Raises:
            ValueError: If any of them still holds its placeholder, naming them all.
        """
        urls = {"url_public": self.url_public}
        if self.route == "s3":
            urls |= {
                "s3.url_endpoint": self.s3.url_endpoint,
                "s3.url_stac_api": self.s3.url_stac_api,
            }
        if unset := [name for name, url in urls.items() if url.startswith("YOUR-")]:
            raise ValueError(
                f"Publishing needs real values for {unset} (pass them with -o "
                f"key=value or a local config), or run with --no-publish."
            )


# ----------------------------------
# Schema registry
# ----------------------------------
SCHEMA_REGISTRY: dict[str, Type[BaseModel]] = {
    "adme": AdmeConfig,
    "aemo": AemoConfig,
    "aeso": AesoConfig,
    "cen": CenConfig,
    "eat": EatConfig,
    "eia": EiaConfig,
    "entsoe": EntsoeConfig,
    "epias": EpiasConfig,
    "ieso": IesoConfig,
    "ons": OnsConfig,
    "rei": ReiConfig,
    "barra2": Barra2Config,
    "era5": Era5Config,
    "icon_dream_eu": IconDreamEuConfig,
    "icon_dream_global": IconDreamGlobalConfig,
    "coordinates": CoordinatesConfig,
    "publish": PublishConfig,
}
