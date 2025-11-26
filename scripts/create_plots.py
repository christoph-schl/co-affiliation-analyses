#!/usr/bin/env python3
from __future__ import annotations

import argparse

from maa.cli.utils import add_standard_cli_arguments, parse_and_configure
from maa.plot.plot import create_plots_from_config


def main() -> None:
    """
    Generate all plots defined in the project configuration.

    Loads plotting settings from the configuration file, validates input
    paths if requested, and produces all plots specified in the config.
    The generated visualizations are written to the configured output
    directory.
    """
    parser = argparse.ArgumentParser(description="Generate all configured plots from project data.")

    # Add shared CLI flags: --config, --validate-paths, --debug
    add_standard_cli_arguments(parser)

    # Parse arguments and configure logging
    args = parse_and_configure(parser)

    # Execute plotting workflow
    create_plots_from_config(
        config_path=args.config,
        validate_paths=args.validate_paths,
        write_outputs_to_file=True,
    )


if __name__ == "__main__":
    main()
