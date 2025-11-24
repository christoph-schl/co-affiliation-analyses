from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import click
import structlog

from maa.cli.utils import (
    NetworkResult,
    iter_year_gaps,
    load_inputs_from_config,
    write_outputs,
)
from maa.config.constants import CONFIGURATION_PATH, ProcessingStage
from maa.config.models import NetworkConfig
from maa.constants.constants import NETWORK_COUNTRY
from maa.network.network import AffiliationNetworkProcessor

_logger = structlog.getLogger(__name__)


def get_network_for_year_gaps(
    article_df: Any, affiliation_gdf: Any, net_cfg: NetworkConfig
) -> Generator[NetworkResult, None, None]:
    """
    Build affiliation networks for each configured year-gap variant.

    This includes:
      • the complete dataset ("all"), and
      • the stable co-affiliation variant ("stable"),
    as defined in the network configuration.

    :param article_df:
        DataFrame containing article metadata.
    :param affiliation_gdf:
        GeoDataFrame containing affiliation information.
    :param net_cfg:
        NetworkConfig object defining year-gap parameters and paths.
    :Yields:
        YearGapResult:
            An object containing:
                • suffix: the variant name ("all", "stable", ...)
                • graph: the constructed affiliation graph
                • link_gdf: the GeoDataFrame of computed affiliation links
    """

    processor = AffiliationNetworkProcessor(
        article_df=article_df,
        affiliation_gdf=affiliation_gdf,
        country_filter=NETWORK_COUNTRY,
    )

    for yg in iter_year_gaps(net_cfg.year_gap_stable_links):
        _logger.info("processing.year_gap", gap=yg.gap, suffix=yg.suffix)
        link_gdf = processor.get_affiliation_links(min_year_gap=yg.gap)
        graph = processor.get_affiliation_graph()
        yield NetworkResult(suffix=yg.suffix, graph=graph, link_gdf=link_gdf)


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
    """CLI entry point for creatin plots from affiliaton networks."""
    create_networks_from_config(
        config_path=config, debug=debug, validate_paths=validate_paths, write_outputs_to_file=True
    )


def create_networks_from_config(
    config_path: Path,
    debug: bool = False,
    validate_paths: bool = False,
    write_outputs_to_file: bool = False,
) -> Generator[NetworkResult, None, None]:
    """
    Build affiliation networks for each configured year-gap variant.

    This function loads input data from the provided configuration file,
    constructs affiliation networks for each defined year-gap variant
    (including the full dataset and the stable co-affiliation variant),
    and optionally writes the results to the configured output directory.

    :param config_path:
        Path to the configuration file specifying input data locations,
        processing settings, and output paths.
    :param debug:
        Enable verbose logging for troubleshooting or development.
    :param validate_paths:
        Validate that input and output paths exist before running.
    :param write_outputs_to_file:
        If True, write all generated network artifacts to disk; if False,
        the networks are generated but not persisted.
    :returns:
        None. Results are optionally written to disk based on configuration.
    """

    input_data = load_inputs_from_config(
        config=config_path,
        stage=ProcessingStage.PREPROCESSING.value,
        validate_paths=validate_paths,
        debug=debug,
    )
    _logger.info("network.build.start", output=str(input_data.config.output_path))
    results = get_network_for_year_gaps(
        article_df=input_data.articles,
        affiliation_gdf=input_data.affiliations,
        net_cfg=input_data.config,
    )

    if write_outputs_to_file:
        write_outputs(results=results, output_path=input_data.config.output_path)
    _logger.info("network.build.done")

    return results


if __name__ == "__main__":
    main()
