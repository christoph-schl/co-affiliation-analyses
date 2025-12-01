# Copyright © 2025 Christoph Schlager, TU Wien

from __future__ import annotations

from pathlib import Path

import click
import structlog

from maa.config.constants import CONFIGURATION_PATH
from maa.config.utils import LogLevel, configure_logging
from maa.network.network import create_top_performers_networks_from_config

_logger = structlog.getLogger(__name__)


@click.command(
    name="create-top-performers-network",
    help="Build affiliation networks for top-performing research organisations.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(path_type=Path),
    default=CONFIGURATION_PATH,
    show_default=True,
    help="Path to config.toml.",
)
@click.option(
    "--validate-paths",
    is_flag=True,
    help="Validate that required paths exist before execution.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable verbose logging.",
)
def main(config: Path, validate_paths: bool, debug: bool) -> None:
    """
    CLI entry point for building networks for top-performing organisations.

    :param config: Path to the configuration file.
    :param validate_paths: Whether to validate input/output paths before execution.
    :param debug: Enable verbose debug logging.
    """
    configure_logging(LogLevel.DEBUG if debug else LogLevel.INFO)

    create_top_performers_networks_from_config(
        config_path=config,
        validate_paths=validate_paths,
        write_outputs_to_file=True,
    )


if __name__ == "__main__":
    main()
