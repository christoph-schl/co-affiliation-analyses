#!/usr/bin/env python3
from __future__ import annotations

import argparse

from maa.cli.utils import add_standard_cli_arguments, parse_and_configure
from maa.network.network import create_networks_from_config


def main() -> None:
    """
    Generate affiliation networks for all configured variants.

    Reads network settings from the configuration file, builds
    year-gap–specific, full-range, and stable co-affiliation
    networks, and writes outputs when enabled.
    """

    parser = argparse.ArgumentParser(
        description="Build affiliation networks for all configured year-gap variants."
    )
    add_standard_cli_arguments(parser=parser)
    args = parse_and_configure(parser=parser)

    create_networks_from_config(
        config_path=args.config,
        validate_paths=args.validate_paths,
        write_outputs_to_file=True,
    )


if __name__ == "__main__":
    main()
