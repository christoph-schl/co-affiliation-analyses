#!/usr/bin/env python3
# Copyright © 2025 Christoph Schlager, TU Wien

from __future__ import annotations

from pathlib import Path

import click

from maa.config.constants import CONFIGURATION_PATH
from maa.config.utils import LogLevel, configure_logging
from maa.routing.routing import create_and_enrich_edges_from_config


@click.command(
    name="enrich-edges",
    help="Build affiliation edges for all unique affiliation pairings and enrich them with "
    "travel data.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(path_type=Path),
    default=CONFIGURATION_PATH,
    show_default=True,
    help="Path to configuration file.",
)
@click.option(
    "--validate-paths", is_flag=True, help="Validate input and output paths before running."
)
@click.option("--debug", is_flag=True, help="Enable verbose debug logging.")
@click.option(
    "--override-edges",
    is_flag=True,
    help="Override existing generated edges if output file already exists.",
)
def main(
    config: Path,
    validate_paths: bool,
    debug: bool,
    override_edges: bool,
) -> None:
    """CLI entry point for building and enriching affiliation edges."""
    configure_logging(LogLevel.DEBUG if debug else LogLevel.INFO)
    create_and_enrich_edges_from_config(
        config_path=config,
        validate_paths=validate_paths,
        write_outputs_to_file=True,
        override_edges=override_edges,
    )


if __name__ == "__main__":
    main()
