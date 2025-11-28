# Copyright © 2025 Christoph Schlager, TU Wien

from __future__ import annotations

from pathlib import Path

import click
import structlog

from maa.config.constants import CONFIGURATION_PATH
from maa.config.utils import write_default_toml

_logger = structlog.getLogger(__name__)


@click.command(name="create-default-config", help="Generate a default config.toml file.")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=CONFIGURATION_PATH,
    show_default=True,
    help="Where to write the TOML file.",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files.")
def main(output: Path, force: bool) -> None:
    """CLI entry point for creating a default TOML configuration."""

    try:
        write_default_toml(path=output, force=force)
    except FileExistsError:
        click.echo(
            f"Error: {output} already exists. Use --force to overwrite.",
            err=True,
        )
        raise SystemExit(1)

    _logger.info("config.write.done")


if __name__ == "__main__":
    main()
