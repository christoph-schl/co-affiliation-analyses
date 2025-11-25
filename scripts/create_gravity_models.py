#!/usr/bin/env python3
from __future__ import annotations

import argparse

from maa.cli.utils import add_standard_cli_arguments, parse_and_configure
from maa.znib.znib import create_znib_gravity_models_from_config


def main() -> None:
    """
    Build ZNIB gravity model inputs and fit all configured model variants.

    Loads parameters from the configuration file, validates paths if
    requested, constructs ZNIB gravity model input datasets, and
    performs the complete model-fitting workflow. Outputs are written
    to the configured directories.
    """
    parser = argparse.ArgumentParser(description="Build inputs and fit ZNIB gravity models.")

    # Add shared CLI flags: --config, --validate-paths, --debug
    add_standard_cli_arguments(parser)

    # Parse arguments and configure logging
    args = parse_and_configure(parser)

    # Execute workflow
    create_znib_gravity_models_from_config(
        config_path=args.config,
        debug=args.debug,
        validate_paths=args.validate_paths,
        write_outputs_to_file=True,
    )


if __name__ == "__main__":
    main()
