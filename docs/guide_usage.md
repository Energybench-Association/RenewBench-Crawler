# Guide: Usage

The CLI scripts in the `scripts` folder coordinate different implemented crawler
functionalities.

## Prerequisites

1. Install the `rbc` package:
```
pip install . [-e]
```
2. For sources that require API tokens or credentials (see the [data sources
   catalogue](data_sources.md)), make sure you have the necessary information at hand.

## Downloading

For downloading data from a source, use scripts ending in `..._download.py`.
For example, you can do:
```commandline
python -m scripts.energy.entsoe_download
```
By default, each script expects a config file located at
`configs/<type>/<source>.yaml` (e.g. `configs/energy/entsoe.yaml`).
Required values can be inserted there.

The scripts are also designed as CLIs, so you can provide user arguments via flags.
It is possible to overwrite the YAML config values via commandline, for example:
```commandline
python -m scripts.energy.entsoe_download -o paths.dst_dir_raw=/my/new/path/
```
For more information, use `--help`.

## Processing

Processing turns a source's downloaded raw data into the data energybench serves. Sources
with a processor have a script `scripts/energy/<source>_process.py`. One run of this processor
performs the following steps:

1. **Coordinates**: locates each energy generating entity (EGE), from the source's own
   locations where it has any, otherwise via [coordinate finding](coordinate_finding.md).
2. **Time**: converts the source's timestamps to UTC, keeping every time step (how each
   source reports time, incl. daylight saving, is in the
   [data sources catalogue](data_sources.md)).
3. **Data**: writes the generation into the source's versioned
   [Icechunk](https://icechunk.io) repository as `values[entity_id, time]` in MW, then
   commits and tags it with the release. Each temporal resolution of the raw data (its
   raw directory, e.g. `1h` or `15min`) gets its own repository,
   `energy/<source>/<resolution>/repository`, with the resolution as an ISO 8601 duration
   (e.g. `PT1H`, `PT15M`) and a time axis at that step.
4. **Catalog**: describes the release as a STAC Collection with one Item per EGE, maps the
   source's fuel types and statuses to the
   [energy extension](https://github.com/Energybench-Association/stac-extension-energy)'s
   vocabulary, validates everything against STAC and each extension used (this fetches
   the official schemas) and publishes it to the STAC API.

Processing stops on any inconsistency instead of skipping data, e.g. a time step that
can't be placed in UTC exactly once, totals changing during reshaping, or a fuel type or
status without a mapping. Fix the cause (or add the mapping) and run again.

### Running

```commandline
python -m scripts.energy.<source>_process
```
Each script reads three configs: the source's (`configs/energy/<source>.yaml`), the
coordinate finding's (`configs/coordinates/coordinates.yaml`) and the publishing one
(`configs/publish/publish.yaml`). Override any value with one `-o` per value:
```commandline
python -m scripts.energy.<source>_process -o paths.dst_dir_raw=/my/raw/path/ -o paths.dst_dir_proc=/my/proc/path/
```
For more information, use `--help`.

If no `dst_dir_proc` is provided, it is derived from `dst_dir_raw`: its `/raw/` part is
replaced with `/processed/` (e.g. `data/raw/<source>/` → `data/processed/<source>/`).
Without one, a `processed/<source>` folder is used next to it.

As this repository is public, its configs hold placeholders where secrets or private
endpoints belong. Pass the real values with `-o`, or keep them in local copies of the
configs that are never committed:

- **Source access** (`access.*` in the source's config): needed whenever the config
  defines it, even though some processors use it only to fetch what isn't cached yet.
- **Publishing URLs** (`data_base_url`, `storage.endpoint`, `stac_api`): publishing refuses
  to start while any of them is still a placeholder.
- **S3 credentials**: read from the `S3_ACCESS_KEY` and `S3_SECRET_KEY` environment
  variables only, never from a config.

To process without publishing, add `--no-publish`: the repositories are written into
`<dst_dir_proc>/icechunk/` and the STAC records only as files, so none of the publishing
URLs or S3 credentials are needed.

### Publishing to energybench

The publishing config (`configs/publish/publish.yaml`) defines:

- `release`: the version written and published, e.g. `v0.1.0`. It becomes the
  repositories' tag and the Collection's suffix (`<source>__<release>`). Tags are
  immutable: running the same release again reuses its tag and writes no data, so raise
  the release to publish new data.
- `data_base_url`: the permanent public data URL. Readers never see the storage: every
  STAC asset's `href` is `{data_base_url}/{bucket}/{prefix}`.
- `storage`: the S3 endpoint and bucket the repositories are written to, and
  `s3_region`, the region S3 requests are signed for. This must be the s3 regions expected
   by the S3 backend (e.g. `us-east-1`).
- `stac_api`: the STAC API the records are published to.
- `chunks`: the arrays' chunk shape, in EGEs and days of data (whatever the resolution).

For example, publishing release `v0.2.0`:
```commandline
python -m scripts.energy.<source>_process -o release=v0.2.0 -o data_base_url=<data_base_url> -o storage.endpoint=<s3_endpoint> -o stac_api=<stac_api_url>
```
A release's new objects are reachable through the data URL only once reception has a new
batch of temporary links for them (delivered by the LSDF). The run logs a reminder
whenever it wrote new data.

### Outputs

| Output | Where |
|---|---|
| Icechunk repositories | published: S3 (`storage`), prefix `energy/<source>/<resolution>/repository`; with `--no-publish`: `<dst_dir_proc>/icechunk/` |
| STAC Collection and Items | `<dst_dir_proc>/stac/<source>__<release>/` (a static catalog), and the STAC API when published |
| Rows a processor keeps out of the data for review (s. the source in the [data sources catalogue](data_sources.md)) | `<dst_dir_proc>/` |
| Coordinate finding results and map | `<dst_dir_proc>/coordinates/` |
| Logs | `<dst_dir_proc>/logs/` |
