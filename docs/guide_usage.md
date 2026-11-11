# Guide: Usage

The files in the `scripts` folder coordinate different implemented crawler functionalities.
Paths and required access information are defined in associated config YAML files in the
[configs](../configs/) folder. As this repository is public, its configs hold
placeholders where secrets or private endpoints belong.

- `.py` files are the CLIs. Define parameters in the local configs or overwrite them via
  the `-o` setting alongside other flags.
- `.sbatch` files run the `.py` files on HPC systems. Define parameters in the local
  configs or set the script's variables as environment variables before running it.

## Prerequisites

1. Install the `rbc` package:
   ```
   pip install [-e] .
   ```
2. For sources that require API tokens or credentials (see the [data sources
   catalogue](data_sources.md)), make sure you have the necessary information at hand.

## Downloading

Downloading parses a source's available online data to download the relevant raw energy /
weather data. The methods involved are unique to each source (e.g. reading excels, making
API requests, etc.).

### Running

For downloading data from a source, use scripts ending in `..._download.py`.

Flags include:
- `--no-resume`: Set if you don't want a run to resume from wherever it left off (if it
  was previously run)

Required YAML configs are:
<details>
<summary><b>The source-specific config</b> (<code>configs/&lt;type&gt;/&lt;source&gt;.yaml</code>)</summary>

- `paths.dst_dir_raw`: Destination directory for storing the raw data
- `access.api_key`: Where required, the API key for accessing the source's data.

</details>

For example, you can do:
```commandline
python -m scripts.energy.<source>_download
```

Override config values via the commandline via:
```commandline
python -m scripts.energy.<source>_download -o paths.dst_dir_raw=/my/new/path/

# For one that needs an API key:
python -m scripts.energy.<source>_download -o paths.dst_dir_raw=/my/new/path/ -o access.api_key=my_api_key
```
For more information, use `--help`.

## Processing

Processing turns a source's downloaded raw data into the data EnergyBench serves. This
performs the following steps:

1. **Coordinates**: locates each energy generating entity (EGE), from the source's own
   locations where it has any, otherwise via [coordinate finding](coordinate_finding.md).
2. **Time**: converts the source's timestamps to UTC, keeping every time step (how each
   source reports time is defined in the [data sources catalogue](data_sources.md)).
3. **Data**: writes the generation into the source's versioned
   [Icechunk](https://icechunk.io) repository as `values[entity_id, time]` in MW, then
   commits and tags it with the release. Each temporal resolution of the raw data (its
   raw directory, e.g. `1h` or `15min`) gets its own repository,
   `energy/<source>/<resolution>/repository`, with the resolution as an ISO 8601 duration
   (e.g. `PT1H`, `PT15M`) and a time axis at that step.
4. **STAC Catalog**: describes the release as a STAC Collection with one Item per EGE,
   defining the source's fuel types and statuses as per the
   [energy extension](https://github.com/EnergyBench-Association/stac-extension-energy),
   then validates and publishes the records (via the route set in the publishing config).

Processing stops on any inconsistency instead of skipping data, e.g. a time step that
can't be placed in UTC exactly once, totals changing during reshaping, or a fuel type or
status without a mapping. Fix the cause (or add the mapping) and run again.

### Running

For processing data from a source, use python scripts ending in `..._process.py` or, on HPC
systems, bash scripts ending in `.sbatch`.

Flags include:
- `--no-publish`: Set if the data shouldn't be published and instead stored to a local
  folder (`<dst_dir_proc>/icechunk/`), meaning none of the publishing URLs or S3
  credentials are needed.

Required YAML configs are:
<details>
<summary><b>The source-specific config</b> (<code>configs/&lt;type&gt;/&lt;source&gt;.yaml</code>)</summary>

- `paths.dst_dir_raw`: Directory containing the raw data.
- `paths.dst_dir_proc`: Destination directory for interim processing results.
  If no `dst_dir_proc` is provided, it is derived from `dst_dir_raw`: its `/raw/` part is
  replaced with `/processed/` (e.g. `data/raw/<source>/` → `data/processed/<source>/`).
- `access.api_key`: Where required, the API key for accessing additional source
  data for enhancing coordinate finding.

</details>
<details>
<summary><b>The coordinate finding config</b> (<code>configs/coordinates/coordinates.yaml</code>)</summary>

- `paths.resources_dir`: Directory containing data for location finding (downloaded from GEM,
  PPM, OSMPP, the OSM API, etc.)

</details>
<details>
<summary><b>The publishing config</b> (<code>configs/publish/publish.yaml</code>)</summary>

- `route`: how the data is published, `lsdf` or `s3`, each with its own block below.
  Currently only the LSDF route can be used; the S3 route is the more direct one planned.
- `release`: the version written and published, e.g. `v0.1.0`. It becomes the
  repositories' tag and the Collection's suffix (`<source>-<tres>__<release>`). Raise
  the release to publish new data.
- `bucket`: the bucket the repositories are written to, on either route.
- `url_public`: the permanent public data URL. Every published STAC asset's `href` is
  `{url_public}/{bucket}/{prefix}`, on either route. Only used when publishing: with
  `--no-publish`, the `href`s name the local repository.
- `lsdf`: the LSDF route, for a cluster that mounts the LSDF. `dst_dir_data` is the
  directory holding the bucket's directory: repositories go to
  `<dst_dir_data>/<bucket>/<prefix>`. `dst_dir_inbox` is a folder the VM watches: each
  release's records are moved into it once complete, and the VM publishes them and signs
  links for the new data within minutes.
- `s3`: the S3 route. `url_endpoint` is the S3 endpoint the repositories are written to,
  and `url_stac_api` the root of the STAC API the records are published to.
- `chunks`: the arrays' chunk shape, in EGEs and days of data (whatever the resolution).

</details>

A release's new objects are reachable through the data URL only once new temporary
links have been signed for them. The run logs a reminder whenever it wrote new data.

#### Via Python scripts

```commandline
python -m scripts.energy.<source>_process
```

Override config values via the commandline via:
```commandline
# Publishing via the `lsdf` route
python -m scripts.energy.<source>_process \
  -o paths.dst_dir_raw=/my/raw/path/ \
  -o paths.dst_dir_proc=/my/proc/path/ \
  -o release=v0.2.0 \
  -o route=lsdf

# Publishing via the `s3` route
python -m scripts.energy.<source>_process \
  -o paths.dst_dir_raw=/my/raw/path/ \
  -o paths.dst_dir_proc=/my/proc/path/ \
  -o release=v0.2.0 \
  -o route=s3 \
  -o s3.url_endpoint=<s3_endpoint_url> \
  -o s3.url_stac_api=<stac_api_url>
```

NOTE: The S3 route needs the `S3_ACCESS_KEY` and `S3_SECRET_KEY` environment variables
to be set!

For more information, use `--help`.

#### Via the sbatch scripts

```commandline
bash scripts/<type>/<source>_process_lsdf.sbatch
```

Override the variables at the top of the script by setting them beforehand:

```commandline
export <SOURCE>_API_KEY=my-key  # if required

RELEASE=v0.2.0 bash scripts/<type>/<source>_process_lsdf.sbatch
```

### Outputs

| Output                                                                                                             | Where                                                                                                                                                                       |
|--------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Processed data (Icechunk repositories)                                                                             | published: `<lsdf.dst_dir_data>/<bucket>/` (LSDF route) or S3 (S3 route), prefix `energy/<source>/<resolution>/repository`; with `--no-publish`: `<dst_dir_proc>/icechunk/` |
| Processed metadata (STAC Collection and Items)                                                                     | `<dst_dir_proc>/stac/<source>-<tres>__<release>/` (a static catalog), and when published: `lsdf.dst_dir_inbox` (LSDF route) or the STAC API (S3 route)                      |
| Rows a processor keeps out of the data for review (s. the source in the [data sources catalogue](data_sources.md)) | `<dst_dir_proc>/`                                                                                                                                                           |
| Coordinate finding results and map                                                                                 | `<dst_dir_proc>/coordinates/`                                                                                                                                               |
| Logs                                                                                                               | `<dst_dir_proc>/logs/`                                                                                                                                                      |
