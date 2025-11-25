from __future__ import annotations

from pathlib import Path

import click
import structlog

from maa.config.constants import CONFIGURATION_PATH
from maa.plot.plot import create_plots_from_config

_logger = structlog.getLogger(__name__)


@click.command(name="create-network", help="Build plots from configuration.")
@click.option(
    "--config",
    "-c",
    type=click.Path(path_type=Path),
    default=CONFIGURATION_PATH,
    show_default=True,
    help="Path to config.toml",
)
@click.option(
    "--stage",
    "-s",
    show_default=True,
    help="Stage group defined in the config file.",
)
@click.option("--validate-paths", is_flag=True, help="Validate paths exist before running.")
@click.option("--debug", is_flag=True, help="Enable verbose logging.")
def main(config: Path, stage: str, validate_paths: bool, debug: bool) -> None:
    """CLI entry point for building affiliation networks."""

    create_plots_from_config(
        config_path=config, debug=debug, validate_paths=validate_paths, write_outputs_to_file=True
    )


if __name__ == "__main__":
    main()
