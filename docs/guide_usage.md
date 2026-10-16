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

For processing data downloaded from a source, use scripts ending in `..._process.py`.
For example, you can do:
```commandline
python -m scripts.energy.cen_process
```
By default, each script expects a config file located at
`configs/<type>/<source>.yaml` (e.g. `configs/energy/cen.yaml`) and the coordinates config at
`configs/coordinates/coordinates.yaml`. Required values can be inserted there.

The scripts are also designed as CLIs, so you can provide user arguments via flags.
It is possible to overwrite the YAML config values via commandline, with one `-o` per value,
for example:
```commandline
python -m scripts.energy.cen_process -o paths.dst_dir_raw=/my/raw/path/ -o paths.dst_dir_proc=/my/proc/path/
```
For more information, use `--help`.

If no `dst_dir_proc` is provided, it is derived from `dst_dir_raw`: its `/raw/` part is
replaced with `/processed/` (e.g. `data/raw/cen/` → `data/processed/cen/`). Without one, a
`processed/<source>` folder is used next to it (e.g. `data/cen/` → `data/processed/cen/`).
