#!/usr/bin/env python3
from __future__ import annotations

import argparse

from maa.cli.utils import add_standard_cli_arguments, parse_and_configure
from maa.routing.routing import create_and_enrich_edges_from_config


def main() -> None:
    """
    Generate affiliation edges and enrich them with travel data.

    Loads network settings from the configuration file, constructs an unfiltered
    co-affiliation network, generates all possible affiliation pairings, and augments
    those edges with corresponding travel data using Valhalla routing engine.
    """
    parser = argparse.ArgumentParser(
        description="Build affiliation edges for all unique affiliation pairings and enrich them"
        " with travel data."
    )

    # adds argument to the default parser args
    parser.add_argument(
        "--override-edges",
        action="store_true",
        default=False,
        help="Override existing generated edges.",
    )

    add_standard_cli_arguments(parser=parser)
    args = parse_and_configure(parser=parser)

    create_and_enrich_edges_from_config(
        config_path=args.config,
        validate_paths=args.validate_paths,
        write_outputs_to_file=True,
        override_edges=args.override_edges,
    )


if __name__ == "__main__":
    main()
