# Copyright © 2025 Christoph Schlager, TU Wien

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, OrderedDict, Union

import geopandas as gpd
import pandas as pd
import structlog
from matplotlib import pyplot as plt

from maa.config.constants import ProcessingStage
from maa.config.loader import load_inputs_from_config
from maa.config.models.input import ImpactConfig, LoadedPlotInputs
from maa.config.models.output import CoAffiliationNetworks, PlotResult, write_outputs
from maa.constants.constants import (
    AFFILIATION_CLASS_COLUMN,
    CO_AFF_ALL_DATASET_NAME,
    CO_AFF_STABLE_DATASET_NAME,
    MWPR_COLUMN,
    ORG_TYPE_FILTER_LIST,
    ORGANISATION_TYPE_COLUMN,
    PREFERRED_AFFILIATION_NAME_COLUMN,
)
from maa.impact.impact import Impact
from maa.impact.utils import (
    AffiliationMwpr,
    AffiliationType,
    aggregate_mwpr_over_time,
    compute_mwpr_for_affiliation_class,
    merge_impact_measures_to_nodes,
)
from maa.network.network import get_network_for_year_gaps
from maa.plot.configuration import PLOT_CONFIGS
from maa.plot.grid import PlotGrid
from maa.plot.utils import (
    add_text_labels_to_bars,
    apply_affiliation_aliases,
    plot_bar,
    plot_time_series,
    plot_violine,
)
from maa.utils.utils import filter_organization_types, get_link_nodes

_logger = structlog.getLogger(__name__)

_COLOR_PALETTE: Dict[str, str] = {
    AffiliationType.AA.value: "#696969",  # grey
    AffiliationType.FA.value: "#1f77b4",  # blue
    AffiliationType.LA.value: "#ff7f0e",  # orange
}
_N_GROUPS_ORG_TYPE = 1000
_TIME_FREQUENCY_YEARS = 1
_N_SAMPLES_PER_GROUP_COLUMN = "n_samples_per_group"


def _add_org_type_subscript_to_group_column(df: pd.DataFrame, group_column: str) -> None:

    df[group_column] = df.apply(
        lambda x: f"{x[group_column]} $^{{({x[ORGANISATION_TYPE_COLUMN][0].upper()})}}$", axis=1
    )


def _build_ordered_label_handle(
    ax1: plt.Axes, ax2: plt.Axes, preferred_order: List[str]
) -> Mapping[str, Any]:

    # get label->handle from each axis
    dict1 = dict(zip(*ax1.get_legend_handles_labels()))
    dict2 = dict(zip(*ax2.get_legend_handles_labels()))

    # prefer dict1 values, then dict2
    merged = dict1.copy()
    merged.update({k: v for k, v in dict2.items() if k not in merged})

    # order by preferred_order then append leftovers
    ordered = OrderedDict()
    for name in preferred_order:
        if name in merged:
            ordered[name] = merged[name]

    for k, v in merged.items():
        if k not in ordered:
            ordered[k] = v

    return ordered


def _add_additional_org_type_legend(ax: plt.Axes) -> None:
    ax.text(
        0.6,
        2.8,
        "(R): Research Institutes    (U): Universities",
        fontsize=14,
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "lightgrey",
            "boxstyle": "round,pad=0.5,rounding_size=0.2",
        },
    )


def _get_y_order(mwpr: pd.DataFrame, x_column: str, y_column: str) -> List[str]:
    return (
        mwpr.loc[
            mwpr[AFFILIATION_CLASS_COLUMN] == AffiliationType.AA.value, [y_column, x_column]
        ].sort_values(by=x_column, ascending=False)
    )[y_column].tolist()


@dataclass(kw_only=True)
class ImpactPlot(Impact):
    filtered_link_gdf: Union[gpd.GeoDataFrame, pd.DataFrame]
    n_mwpr_units: int = 10
    _filtered_node_df: Optional[pd.DataFrame] = field(default=None, init=False)

    _mwpr_df: Dict[str, AffiliationMwpr] = field(default_factory=dict, init=False)
    _mwpr_filtered_df: Dict[str, AffiliationMwpr] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        # filter organisation types
        if self.allowed_org_types is not None:
            self.filtered_link_gdf = filter_organization_types(
                df=self.filtered_link_gdf, org_types=self.allowed_org_types
            )

        self._filtered_node_df = get_link_nodes(link_gdf=self.filtered_link_gdf)
        self._filtered_node_df = merge_impact_measures_to_nodes(
            node_df=self._filtered_node_df, impact_df=self.impact_df
        )

        # apply affiliation aliases
        for df in [self._node_df, self._filtered_node_df]:
            apply_affiliation_aliases(df=df, column=PREFERRED_AFFILIATION_NAME_COLUMN)

    @property
    def filtered_node_df(self) -> Optional[pd.DataFrame]:
        return self._filtered_node_df

    def get_bar_plot(
        self,
        min_samples: int,
        group_column: str = PREFERRED_AFFILIATION_NAME_COLUMN,
        n_groups: int = _N_GROUPS_ORG_TYPE,
    ) -> PlotGrid:
        """
        Generates a grouped bar plot showing mwPR values
        for different affiliation classes (all, first, or the last affiliation),
        using both the full and filtered node datasets.

        The grouping is performed using the `preferred_name` column, which identifies. The mwPR
        values are calculated separately for the full dataset and a filtered subset, allowing
        side-by-side comparison in the resulting bar plot.

        :param min_samples: Minimum number of samples required for a group to be included.
        :param group_column: Column name used to group nodes (e.g., "org_type" or "preferred_name").
        :param n_groups: Number of top groups to include in the plot.
        :return: A PlotGrid object containing the figure and two subplot axes.
        """

        self._compute_mwprs(group_column=group_column, min_samples=min_samples, n_groups=n_groups)

        # plot onto the pre-prepared grid axes
        grid = self._plot_bars_onto_grid(group_column=group_column)

        return grid

    def get_timeseries_plot(
        self,
        group_column: str,
        min_samples: int = 0,
        n_groups: int = _N_GROUPS_ORG_TYPE,
        time_freq_years: int = _TIME_FREQUENCY_YEARS,
    ) -> PlotGrid:
        """
        Generate a time series plot for the specified group column.

        :param group_column: Column name used to group nodes (e.g., "org_type" or "preferred_name").
        :param min_samples: Minimum number of samples required for a group to be included.
        :param n_groups: Number of top groups to include in the plot.
        :param time_freq_years: Time interval in years for grouping.
        :return: A PlotGrid object containing the figure and two subplot axes.
        """

        self._compute_mwprs(group_column=group_column, min_samples=min_samples, n_groups=n_groups)
        grid = self._plot_time_series_onto_grid(
            node_df=self._node_df,
            node_df_filtered=self._filtered_node_df,
            group_column=group_column,
            time_freq_years=time_freq_years,
        )
        return grid

    def get_violine_plot(self, group_column: str = ORGANISATION_TYPE_COLUMN) -> PlotGrid:
        """
        Plot violin distributions of Hazen percentiles for each organisation type and
        annotate the plot with mwPR markers for 'all', 'first' and 'last' groups.

        :param group_column: Column name used to group nodes (e.g., "org_type" or "preferred_name").
        :return: A PlotGrid containing the completed violine plot visualization.
        """

        self._compute_mwprs(group_column=group_column)

        grid = self._plot_violins_onto_grid(group_column)

        return grid

    def _plot_violins_onto_grid(self, group_column: str) -> PlotGrid:
        # preserve a stable ordering for the categories
        org_type_order = []
        if self._mwpr_df[group_column] is not None:
            org_type_order = list(self._mwpr_df[group_column].all[ORGANISATION_TYPE_COLUMN])
        # get grid from config
        config_key = f"violine_by_{group_column}"
        grid = PlotGrid.from_plot_name(config_key)
        # pair axes with the DataFrames we want to visualise
        axes_and_data = (
            (grid.ax1, self._node_df, self._mwpr_df[group_column]),
            (grid.ax2, self._filtered_node_df, self._mwpr_filtered_df[group_column]),
        )
        for ax, nodes_df, mwpr_df in axes_and_data:
            if mwpr_df is not None:
                plot_violine(
                    ax=ax,
                    nodes_df=nodes_df,
                    mwpr=mwpr_df,
                    org_type_order=org_type_order,
                )
            ax.set_xlabel(None)

        # explicitly required because seaborn inside plot_violine overrides y-labels
        if PLOT_CONFIGS[config_key].figure.deactivate_ax2_ylabels:
            grid.ax2.set_ylabel(None)

        legend_config = PLOT_CONFIGS[config_key].legend
        grid.add_legends_from_plot_config(config=legend_config)
        return grid

    def _plot_bars_onto_grid(
        self,
        group_column: str = PREFERRED_AFFILIATION_NAME_COLUMN,
    ) -> PlotGrid:

        # get grid from config
        config_key = f"barplot_by_{group_column}"
        grid = PlotGrid.from_plot_name(config_key)

        for mwpr, ax in [
            (self._mwpr_df[group_column].to_concatenated(), grid.ax1),
            (self._mwpr_filtered_df[group_column].to_concatenated(), grid.ax2),
        ]:
            if group_column == PREFERRED_AFFILIATION_NAME_COLUMN:
                _add_org_type_subscript_to_group_column(df=mwpr, group_column=group_column)

            y_order = _get_y_order(
                mwpr=mwpr,
                x_column=MWPR_COLUMN,
                y_column=PREFERRED_AFFILIATION_NAME_COLUMN,
            )

            plot_bar(ax=ax, data=mwpr, y_order=y_order)

            add_text_labels_to_bars(
                df=mwpr,
                ax=ax,
                y_column=PREFERRED_AFFILIATION_NAME_COLUMN,
                y_order=y_order,
                fontsize=10,
            )

        _add_additional_org_type_legend(ax=grid.legend_ax)

        legend_config = PLOT_CONFIGS[config_key].legend
        grid.add_legends_from_plot_config(config=legend_config)

        return grid

    def _compute_mwprs(
        self,
        min_samples: int = 0,
        group_column: str = ORGANISATION_TYPE_COLUMN,
        n_groups: int = _N_GROUPS_ORG_TYPE,
    ) -> None:

        self._mwpr_df[group_column] = compute_mwpr_for_affiliation_class(
            node_df=self._node_df,
            n_groups=n_groups,
            group_column=group_column,
            min_samples=min_samples,
        )

        self._mwpr_filtered_df[group_column] = compute_mwpr_for_affiliation_class(
            node_df=self._filtered_node_df,
            n_groups=n_groups,
            group_column=group_column,
            min_samples=min_samples,
        )

    def _plot_time_series_onto_grid(
        self,
        node_df: pd.DataFrame,
        node_df_filtered: pd.DataFrame,
        group_column: str,
        time_freq_years: int = _TIME_FREQUENCY_YEARS,
    ) -> PlotGrid:

        # aggregate nodes over time
        time_nodes = aggregate_mwpr_over_time(
            node_df=node_df, group_column=group_column, time_freq=time_freq_years
        )
        time_nodes_filtered = aggregate_mwpr_over_time(
            node_df=node_df_filtered, group_column=group_column, time_freq=time_freq_years
        )

        # get grid from config
        config_key = f"timeseries_by_{group_column}"
        grid = PlotGrid.from_plot_name(config_key)

        # filter and plot aggregated nodes
        group_list = self._mwpr_df[group_column].groups
        for time_nodes, ax in [(time_nodes, grid.ax1), (time_nodes_filtered, grid.ax2)]:

            # filter top n units (e.g., preferred_name or org_type)
            time_nodes = time_nodes[time_nodes[group_column].isin(group_list)].reset_index(
                drop=True
            )

            if group_column == PREFERRED_AFFILIATION_NAME_COLUMN:
                _add_org_type_subscript_to_group_column(df=time_nodes, group_column=group_column)

            plot_time_series(df=time_nodes, axes=ax, group_column=group_column)

        # build handles for legend
        preferred_order = list(time_nodes[group_column].unique())
        label_handle = _build_ordered_label_handle(
            ax1=grid.ax1, ax2=grid.ax2, preferred_order=preferred_order
        )

        legend_config = PLOT_CONFIGS[config_key].legend
        grid.add_legends_from_plot_config(config=legend_config, label_handle=label_handle)

        return grid


def get_plots_from_networks(
    networks: CoAffiliationNetworks, impact_df: pd.DataFrame, plot_cfg: ImpactConfig
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

    plot = ImpactPlot(
        link_gdf=getattr(networks, CO_AFF_ALL_DATASET_NAME).link_gdf,
        filtered_link_gdf=getattr(networks, CO_AFF_STABLE_DATASET_NAME).link_gdf,
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


def create_plots_from_config(
    config_path: Path, validate_paths: bool = False, write_outputs_to_file: bool = False
) -> PlotResult:
    input_data = load_inputs_from_config(
        config=config_path,
        stage=ProcessingStage.IMPACT.value,
        validate_paths=validate_paths,
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
    assert isinstance(plot_cfg, ImpactConfig)
    assert isinstance(input_data, LoadedPlotInputs)
    plot_results = get_plots_from_networks(
        networks=results, plot_cfg=plot_cfg, impact_df=input_data.impact
    )
    _logger.info("plot.build.done")

    if write_outputs_to_file:
        write_outputs(results=plot_results, output_path=input_data.config.output_path)

    return plot_results
