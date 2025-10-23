from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, OrderedDict

import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt

from src.mma.constants import (
    AFFILIATION_CLASS_COLUMN,
    MWPR_COLUMN,
    ORG_TYPE_FILTER_LIST,
    ORGANISATION_TYPE_COLUMN,
    PREFERRED_AFFILIATION_NAME_COLUMN,
)
from src.mma.impact.impact import Impact
from src.mma.impact.utils import (
    AffiliationMwpr,
    AffiliationType,
    aggregate_mwpr_over_time,
    compute_mwpr_for_affiliation_class,
    merge_impact_measures_to_nodes,
)
from src.mma.plot.configuration import PLOT_CONFIGS
from src.mma.plot.constants import AFFILIATION_NAME_ALIASES
from src.mma.plot.grid import PlotGrid
from src.mma.plot.utils import (
    add_text_labels_to_bars,
    plot_bar,
    plot_time_series,
    plot_violine,
)
from src.mma.utils.utils import filter_organization_types, get_link_nodes

_COLOR_PALETTE: Dict[str, str] = {
    AffiliationType.ALL.value: "#696969",  # grey
    AffiliationType.FIRST.value: "#1f77b4",  # blue
    AffiliationType.LAST.value: "#ff7f0e",  # orange
}
_N_GROUPS_ORG_TYPE = 1000
_TIME_FREQUENCY_YEARS = 1


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


def _get_y_order(mwpr: pd.DataFrame, x_column: str, y_column: str) -> List[str]:
    return (
        mwpr.loc[
            mwpr[AFFILIATION_CLASS_COLUMN] == AffiliationType.ALL.value, [y_column, x_column]
        ].sort_values(by=x_column, ascending=False)
    )[y_column].tolist()


@dataclass
class ImpactPlot(Impact):
    filtered_link_gdf: gpd.GeoDataFrame
    n_mwpr_units: int = 10
    _filtered_node_df: Optional[pd.DataFrame] = field(default=None, init=False)

    _mwpr_df: Dict[str, AffiliationMwpr] = field(default_factory=dict, init=False)
    _mwpr_filtered_df: Dict[str, AffiliationMwpr] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        # filter organisation types
        self.filtered_link_gdf = filter_organization_types(
            df=self.filtered_link_gdf, org_types=ORG_TYPE_FILTER_LIST
        )

        self._filtered_node_df = get_link_nodes(link_gdf=self.filtered_link_gdf)
        self._filtered_node_df = merge_impact_measures_to_nodes(
            node_df=self._filtered_node_df, impact_df=self.impact_df
        )

        # apply affiliation aliases
        for df in [self._node_df, self._filtered_node_df]:
            df[PREFERRED_AFFILIATION_NAME_COLUMN] = df[PREFERRED_AFFILIATION_NAME_COLUMN].apply(
                lambda x: AFFILIATION_NAME_ALIASES.get(x, x)
            )

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

            # remove individual x-labels since the grid controls overall labeling
            ax.set_xlabel(None)
        # Labeling and legend belong to the grid-level axes
        grid.ax1.set_ylabel("Hazen percentile [%]", labelpad=0, fontsize=14)
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
            )

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

            plot_time_series(df=time_nodes, axes=ax, group_column=group_column)
            ax.set_ylim(20, 100)
            ax.legend_.remove()

        # build handles for legend
        preferred_order = list(time_nodes[group_column].unique())
        label_handle = _build_ordered_label_handle(
            ax1=grid.ax1, ax2=grid.ax2, preferred_order=preferred_order
        )

        legend_config = PLOT_CONFIGS[config_key].legend
        grid.add_legends_from_plot_config(config=legend_config, label_handle=label_handle)

        return grid
