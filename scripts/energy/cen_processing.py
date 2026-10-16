#!/usr/bin/env python
"""CEN DATA PROCESSING SCRIPT.

Process raw data previously downloaded from the CEN API portal for Chile.
"""

from argparse import ArgumentParser, Namespace
from pprint import pformat
from typing import cast

from loguru import logger

from rbc.config.loader import load_config, parse_key_value_pairs
from rbc.config.schema import CoordinatesConfig
from rbc.energy.cen.processor import SOURCE, CenProcessor
from rbc.utils import setup_logging


def parse_arguments() -> Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Namespace parsed command line arguments.
    """
    parser = ArgumentParser(prog="CEN data processing into Zarr and STAC")
    # parser.add_argument(
    #     "--no-resume",
    #     dest="resume",
    #     action="store_false",
    #     help="Do not resume processing from a previous checkpoint.",
    # )
    # parser.set_defaults(resume=True)
    parser.add_argument(
        "-o",
        "--cfg_options",
        action="append",
        help=(
            "Override YAML config values (supports nested keys). Example: "
            "-o paths.dst_dir_raw=/your/raw/path/ paths.dst_dir_proc=/your/proc/path/ "
            " access.api_key=YOUR-SECRET-KEY"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Coordinating CEN data processing."""
    args = parse_arguments()
    overrides = parse_key_value_pairs(args.cfg_options) if args.cfg_options else None

    cfg = load_config(source=SOURCE, overrides=overrides)
    coordinates_cfg = cast(
        CoordinatesConfig, load_config(source="coordinates", overrides=overrides)
    )
    setup_logging(output_dir=cfg.paths.dst_dir_proc)

    logger.info(f"Flags for processing:\n{pformat(vars(args), sort_dicts=False)}")
    logger.info(
        f"Config schema for processing:\n{pformat(dict(cfg), sort_dicts=False)}"
    )

    processor = CenProcessor(
        token=cfg.access.api_key,
        input_dir=cfg.paths.dst_dir_raw,
        output_dir=cfg.paths.dst_dir_proc,
        resources_dir=coordinates_cfg.paths.resources_dir,
    )
    processor.process_data()


if __name__ == "__main__":
    main()
