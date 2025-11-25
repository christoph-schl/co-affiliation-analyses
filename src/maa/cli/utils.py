"""
Shared CLI utilities for project scripts based on argparse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from maa.config.constants import CONFIGURATION_PATH
from maa.config.utils import LogLevel, configure_logging


def add_standard_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the shared CLI arguments used across all project scripts.

    Includes:
    - --config / -c: path to config.toml
    - --validate-paths: validate input paths before execution
    - --debug: enable verbose debug logging
    """
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path(CONFIGURATION_PATH),
        help="Path to config.toml (default: project configuration).",
    )
    parser.add_argument(
        "--validate-paths",
        action="store_true",
        help="Validate that all referenced input paths exist before running.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )


def parse_and_configure(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parse command-line arguments and configure logging.

    Returns:
        argparse.Namespace: parsed CLI arguments.
    """
    args = parser.parse_args()

    # Configure logging once arguments are known
    configure_logging(LogLevel.DEBUG if args.debug else LogLevel.INFO)

    return args
