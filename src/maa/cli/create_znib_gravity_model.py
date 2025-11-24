from pathlib import Path
from typing import Generator

import click
import geopandas as gpd
import pandas as pd
import structlog

from maa.cli.create_network import write_outputs
from maa.cli.utils import (
    ZNIBGravityResult,
    iter_year_gaps,
    load_inputs_from_config,
)
from maa.config.constants import CONFIGURATION_PATH, ProcessingStage
from maa.config.models import GravityConfig, LoadedGravityInputs
from maa.constants.constants import NETWORK_COUNTRY, ORG_TYPE_FILTER_LIST
from maa.dataframe.models.route import get_empty_routes_df
from maa.znib.configuration import config_inter_proximity, config_intra_proximity
from maa.znib.znib import ZNIB

_logger = structlog.getLogger(__name__)


def get_gravity_model_for_year_gaps(
    article_df: pd.DataFrame,
    affiliation_gdf: gpd.GeoDataFrame,
    routes_df: pd.DataFrame,
    gravity_cfg: GravityConfig,
) -> Generator[ZNIBGravityResult, None, None]:
    """
    Build ZNIB gravity model input for each configured year-gap variant.

    Generates results for:
      • the complete co-affiliation dataset ("all"), and
      • the stable co-affiliation variant ("stable"),
    as defined in the network configuration.

    :param article_df:
        DataFrame containing article metadata.
    :param affiliation_gdf:
        GeoDataFrame containing affiliation information.
    :param routes_df:
        DataFrame containing travel time information for each affiliation pair.
    :param gravity_cfg:
        GravityConfig object defining year-gap parameters and paths.
    :Yields:
        YearGapResult:
            An object containing:
                • suffix: the variant name ("all", "stable", ...)
                • graph: the constructed affiliation graph
                • link_gdf: the GeoDataFrame of computed affiliation links
    """

    znib = ZNIB(
        article_df=article_df, affiliation_gdf=affiliation_gdf, country_filter=NETWORK_COUNTRY
    )

    for yg in iter_year_gaps(gravity_cfg.year_gap_stable_links):
        znib.min_year_gap = yg.gap
        _logger.info("processing.year_gap", gap=yg.gap, suffix=yg.suffix)
        model_data = znib.enrich_edges_with_org_info(
            route_df=routes_df,
            org_type_list=ORG_TYPE_FILTER_LIST,
        )

        znib_intra_model = None
        znib_inter_model = None
        if gravity_cfg.fit_models:
            _logger.info(
                "fit intra organisational znib gravity model", gap=yg.gap, suffix=yg.suffix
            )
            znib_intra_model = znib.fit_znib(config=config_intra_proximity)
            _logger.info(znib_intra_model.summary)

            _logger.info(
                "fit inter organisational znib gravity model", gap=yg.gap, suffix=yg.suffix
            )
            znib_inter_model = znib.fit_znib(config=config_inter_proximity)
            _logger.info(znib_inter_model.summary)

        yield ZNIBGravityResult(
            suffix=yg.suffix,
            graph=znib.edge,
            link_gdf=znib.link,
            znib_data=model_data,
            znib_intra_model=znib_intra_model,
            znib_inter_model=znib_inter_model,
        )


@click.command(name="create-gravity", help="Build ZNIB gravity model inputs from configuration.")
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
    default=ProcessingStage.GRAVITY.value,
    show_default=True,
    help="Stage group defined in the config file.",
)
@click.option("--validate-paths", is_flag=True, help="Validate paths exist before running.")
@click.option("--dry-run", is_flag=True, help="Do not write any files.")
@click.option("--debug", is_flag=True, help="Enable verbose logging.")
def main(config: Path, stage: str, validate_paths: bool, dry_run: bool, debug: bool) -> None:
    """CLI entry point for creating znib gravity model inputs and fitting the models."""

    input_data = load_inputs_from_config(
        config=config, stage=stage, validate_paths=validate_paths, debug=debug
    )
    gravity_cfg = input_data.config
    article_df = input_data.articles
    affiliation_gdf = input_data.affiliations

    if isinstance(input_data, LoadedGravityInputs):
        routes_df = input_data.routes
    else:
        routes_df = get_empty_routes_df()

    # 🔥type narrowing for mypy:
    assert isinstance(gravity_cfg, GravityConfig)

    results = get_gravity_model_for_year_gaps(
        article_df=article_df,
        affiliation_gdf=affiliation_gdf,
        routes_df=routes_df,
        gravity_cfg=gravity_cfg,
    )

    write_outputs(results=results, output_path=gravity_cfg.output_path, dry_run=dry_run)


if __name__ == "__main__":
    main()
