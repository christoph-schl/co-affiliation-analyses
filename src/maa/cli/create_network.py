from __future__ import annotations

from pathlib import Path

import click
import structlog

from maa.config.constants import CONFIGURATION_PATH
from maa.network.network import create_networks_from_config

_logger = structlog.getLogger(__name__)


@click.command(name="create-network", help="Build affiliation networks from configuration.")
@click.option(
    "--config",
    "-c",
    type=click.Path(path_type=Path),
    default=CONFIGURATION_PATH,
    show_default=True,
    help="Path to config.toml",
)
@click.option("--validate-paths", is_flag=True, help="Validate paths exist before running.")
@click.option("--debug", is_flag=True, help="Enable verbose logging.")
def main(config: Path, validate_paths: bool, debug: bool) -> None:
    """CLI entry point for creating plots from affiliation networks."""
    create_networks_from_config(
        config_path=config, debug=debug, validate_paths=validate_paths, write_outputs_to_file=True
    )


if __name__ == "__main__":
    main()
