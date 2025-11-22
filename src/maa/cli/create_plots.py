from __future__ import annotations

from pathlib import Path
from typing import Generator

import click
import pandas as pd
import structlog

from maa.cli.create_network import get_network_for_year_gaps
from maa.cli.utils import (
    NetworkResult,
    PlotResult,
    load_inputs_from_config,
    write_outputs,
)
from maa.config.constants import CONFIGURATION_PATH, ProcessingStage
from maa.config.models import LoadedPlotInputs, PlotConfig
from maa.constants.constants import (
    CO_AFF_ALL_DATASET_NAME,
    CO_AFF_STABLE_DATASET_NAME,
    ORG_TYPE_FILTER_LIST,
    ORGANISATION_TYPE_COLUMN,
    PREFERRED_AFFILIATION_NAME_COLUMN,
)
from maa.plot.plot import ImpactPlot

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
    default=ProcessingStage.PLOT.value,
    show_default=True,
    help="Stage group defined in the config file.",
)
@click.option("--validate-paths", is_flag=True, help="Validate paths exist before running.")
@click.option("--dry-run", is_flag=True, help="Do not write any files.")
@click.option("--debug", is_flag=True, help="Enable verbose logging.")
def main(config: Path, stage: str, validate_paths: bool, dry_run: bool, debug: bool) -> None:
    """CLI entry point for building affiliation networks."""

    input_data = load_inputs_from_config(
        config=config, stage=stage, validate_paths=validate_paths, debug=debug
    )

    _logger.info("network.build.start", output=str(input_data.config.output_path))
    results = get_network_for_year_gaps(
        article_df=input_data.articles,
        affiliation_gdf=input_data.affiliations,
        net_cfg=input_data.config,
    )
    _logger.info("network.build.done")

    _logger.info("plot.build.start", output=str(input_data.config.output_path))
    plot_cfg = input_data.config

    # 🔥type narrowing for mypy:
    assert isinstance(plot_cfg, PlotConfig)
    assert isinstance(input_data, LoadedPlotInputs)

    plot_results = get_plots_from_networks(
        networks=results, plot_cfg=plot_cfg, impact_df=input_data.impact
    )
    _logger.info("plot.build.done")

    write_outputs(results=plot_results, output_path=input_data.config.output_path, dry_run=dry_run)


def get_plots_from_networks(
    networks: Generator[NetworkResult, None, None], impact_df: pd.DataFrame, plot_cfg: PlotConfig
) -> PlotResult:
    """
     Generate all plots derived from a collection of network analysis results.

        This function consumes a generator of ``NetworkResult`` objects, builds an
        internal mapping from result suffixes to their corresponding link GeoDataFrames,
        and initializes an ``ImpactPlot`` object to compute multiple visualizations
        (violin plot, bar plot, and two time series plots). These plots are packaged
        into a single ``PlotResult`` for downstream saving or rendering.

    :param networks:
        A generator yielding ``NetworkResult`` objects for the `all` and the `stable` dataset
        with co-affiliation links.
    :param impact_df:
        A dataframe containing impact metrics to visualize. Passed directly to
        ``ImpactPlot``.
    :param plot_cfg:
        Configuration specifying plot parameters such as minimum sample thresholds
        and maximum groups for aggregation.
    :return:
        A structured container holding all generated plot objects:
        violin plot, bar plot, time series by organisation type, and time series
        by institution.
    """

    link_map = {result.suffix: result.link_gdf for result in networks}
    plot = ImpactPlot(
        link_gdf=link_map[CO_AFF_ALL_DATASET_NAME],
        filtered_link_gdf=link_map[CO_AFF_STABLE_DATASET_NAME],
        impact_df=impact_df,
        allowed_org_types=ORG_TYPE_FILTER_LIST,
    )
    violine_plot = plot.get_violine_plot()
    bar_plot = plot.get_bar_plot(
        min_samples=plot_cfg.min_samples,
        group_column=PREFERRED_AFFILIATION_NAME_COLUMN,
        n_groups=plot_cfg.max_groups,
    )
    timeseries_plot_org_type = plot.get_timeseries_plot(group_column=ORGANISATION_TYPE_COLUMN)
    timeseries_plot_inst = plot.get_timeseries_plot(
        group_column=PREFERRED_AFFILIATION_NAME_COLUMN,
        min_samples=plot_cfg.min_samples,
        n_groups=plot_cfg.max_groups,
    )
    plot_results = PlotResult(
        violine_plot=violine_plot,
        bar_plot=bar_plot,
        timeseries_plot_org_type=timeseries_plot_org_type,
        timeseries_plot_institution=timeseries_plot_inst,
    )
    return plot_results


if __name__ == "__main__":
    main()
