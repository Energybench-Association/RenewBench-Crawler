#!/usr/bin/env python
"""ONS DATA PROCESSING SCRIPT.

Process raw data previously downloaded from the ONS open data portal for Brazil.
"""

from argparse import ArgumentParser, Namespace
from pprint import pformat
from typing import cast

from loguru import logger

from rbc.config.loader import (
    load_config,
    parse_key_value_pairs,
    parse_period,
    parse_release,
)
from rbc.config.schema import CoordinatesConfig, PublishConfig
from rbc.energy.ons.processor import SOURCE, OnsProcessor
from rbc.utils import setup_logging


def parse_arguments() -> Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Namespace parsed command line arguments.
    """
    parser = ArgumentParser(prog="ONS data processing into Icechunk and STAC")
    parser.add_argument(
        "-r",
        "--release",
        required=True,
        type=parse_release,
        help=(
            "The release to write and publish, as v<major>.<minor>.<patch>. It tags the "
            "repositories and names the STAC Collection. Initial full runs use v0.1.0; "
            "raise it for new data, as a reused release writes nothing new."
        ),
    )
    parser.add_argument(
        "-p",
        "--periods",
        nargs="+",
        type=parse_period,
        help=(
            "Years or months of raw data to process. Example: -p 2018 2020-03. "
            "Default: all raw files. The time axis spans the first to the last month."
        ),
    )
    parser.add_argument(
        "--no-publish",
        dest="publish",
        action="store_false",
        help=(
            "Write the Icechunk repository and the STAC JSON into the processed data "
            "directory only, instead of publishing them."
        ),
    )
    parser.add_argument(
        "-o",
        "--cfg_options",
        action="append",
        help=(
            "Override YAML config values (supports nested keys), one -o per value. "
            "Example: -o paths.dst_dir_raw=/your/raw/path/ "
            "-o paths.dst_dir_proc=/your/proc/path/"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Coordinating ONS data processing."""
    args = parse_arguments()
    overrides = parse_key_value_pairs(args.cfg_options) if args.cfg_options else None

    cfg = load_config(source=SOURCE, overrides=overrides)
    coordinates_cfg = cast(
        CoordinatesConfig, load_config(source="coordinates", overrides=overrides)
    )
    publish_cfg = cast(
        PublishConfig, load_config(source="publish", overrides=overrides)
    )
    setup_logging(output_dir=cfg.paths.dst_dir_proc)

    logger.info(f"Flags for processing:\n{pformat(vars(args), sort_dicts=False)}")
    shown = cfg.model_dump()
    logger.info(f"Config schema for processing:\n{pformat(shown, sort_dicts=False)}")

    processor = OnsProcessor(
        input_dir=cfg.paths.dst_dir_raw,
        output_dir=cfg.paths.dst_dir_proc,
        resources_dir=coordinates_cfg.paths.resources_dir,
        release=args.release,
        publish_cfg=publish_cfg,
        publish=args.publish,
        periods=args.periods,
    )
    processor.process_data()


if __name__ == "__main__":
    main()
