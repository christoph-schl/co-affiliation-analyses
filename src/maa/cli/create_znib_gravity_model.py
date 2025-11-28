# Copyright © 2025 Christoph Schlager, TU Wien

from pathlib import Path

import click
import structlog

from maa.config.constants import CONFIGURATION_PATH
from maa.config.utils import LogLevel, configure_logging
from maa.znib.znib import create_znib_gravity_models_from_config

_logger = structlog.getLogger(__name__)


@click.command(name="create-gravity", help="Build ZNIB gravity model inputs from configuration.")
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
    """CLI entry point for creating znib gravity model inputs and fitting the models."""
    configure_logging(LogLevel.DEBUG if debug else LogLevel.INFO)
    create_znib_gravity_models_from_config(
        config_path=config, validate_paths=validate_paths, write_outputs_to_file=True
    )


if __name__ == "__main__":
    main()
