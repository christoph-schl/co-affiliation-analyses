from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    compute_mwpr_for_affiliation_class,
    merge_impact_measures_to_nodes,
)
from src.mma.plot.constants import AFFILIATION_NAME_ALIASES
from src.mma.plot.utils import (
    Grid,
    add_legend,
    add_text_labels_to_bars,
    get_plotting_grid,
    plot_bar,
    plot_violine,
)
from src.mma.utils.utils import filter_organization_types, get_link_nodes

_COLOR_PALETTE: Dict[str, str] = {
    AffiliationType.ALL.value: "#696969",  # grey
    AffiliationType.FIRST.value: "#1f77b4",  # blue
    AffiliationType.LAST.value: "#ff7f0e",  # orange
}
_N_GROUPS_ORG_TYPE = 1000


def _get_y_order(mwpr: pd.DataFrame, x_column: str, y_column: str) -> List[str]:
    return (
        mwpr.loc[
            mwpr[AFFILIATION_CLASS_COLUMN] == AffiliationType.ALL.value, [y_column, x_column]
        ].sort_values(by=x_column, ascending=False)
    )[y_column].tolist()


@dataclass
class ImpactPlot(Impact):
    filtered_link_gdf: gpd.GeoDataFrame
    min_samples_org_name: int = 300
    n_orgs: int = 10
    _bar_plot_grid: Grid = field(
        default_factory=lambda: get_plotting_grid(
            create_hline=False,
            deactivate_ax2_ylabels=False,
            figure_size=(12, 6),
            x_label="mwPR [%]",
        )
    )
    _violine_plot_grid: Grid = field(default_factory=lambda: get_plotting_grid())
    _filtered_node_df: Optional[pd.DataFrame] = field(default=None, init=False)
    _mwpr_df_org_name: Optional[pd.DataFrame] = field(default=None, init=False)
    _mwpr_filtered_df_org_name: Optional[pd.DataFrame] = field(default=None, init=False)
    _mwpr_df_org_type: Optional[AffiliationMwpr] = field(default=None, init=False)
    _mwpr_filtered_df_org_type: Optional[AffiliationMwpr] = field(default=None, init=False)

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

    def get_bar_plot(self) -> plt.Figure:
        """
        Generates a grouped bar plot showing mwPR values
        for different affiliation classes (all, first, or the last affiliation),
        using both the full and filtered node datasets.

        The grouping is performed using the `preferred_name` column, which identifies. The mwPR
        values are calculated separately for the full dataset and a filtered subset, allowing
        side-by-side comparison in the resulting bar plot.

        :return:
            A matplotlib Figure object containing the completed bar plot visualization.
        """

        self._compute_org_name_mwprs()

        y_order_ax1 = _get_y_order(
            mwpr=self._mwpr_df_org_name,
            x_column=MWPR_COLUMN,
            y_column=PREFERRED_AFFILIATION_NAME_COLUMN,
        )
        y_order_ax2 = _get_y_order(
            mwpr=self._mwpr_filtered_df_org_name,
            x_column=MWPR_COLUMN,
            y_column=PREFERRED_AFFILIATION_NAME_COLUMN,
        )

        # plot onto the pre-prepared grid axes
        self._plot_bars_onto_grid(y_order_ax1=y_order_ax1, y_order_ax2=y_order_ax2)

        # annotate bars with sample counts
        self._add_bar_labels(y_order_ax1=y_order_ax1, y_order_ax2=y_order_ax2)

        add_legend(
            ax=self._bar_plot_grid.ax1,
            fig=self._bar_plot_grid.fig,
            add_all_and_mwpr=False,
            placement="lower center",
            n_legend_columns=3,
            fontsize=14,
            bbox_to_anchor=(0.25, 0.05),
        )

        return self._bar_plot_grid.fig

    def get_violine_plot(self) -> plt.Figure:
        """
        Plot violin distributions of Hazen percentiles for each organisation type and
        annotate the plot with mwPR markers for 'all', 'first' and 'last' groups.

         :return:
            A matplotlib Figure object containing the completed violine plot visualization.
        """

        self._compute_org_type_mwprs()

        # preserve a stable ordering for the categories
        org_type_order = []
        if self._mwpr_df_org_type is not None:
            org_type_order = list(self._mwpr_df_org_type.all[ORGANISATION_TYPE_COLUMN])

        grid = self._violine_plot_grid

        # pair axes with the DataFrames we want to visualise
        axes_and_data = (
            (grid.ax1, self._node_df, self._mwpr_df_org_type),
            (grid.ax2, self._filtered_node_df, self._mwpr_filtered_df_org_type),
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

        add_legend(ax=grid.ax1, fig=grid.fig, n_legend_columns=4)

        return grid.fig

    def _add_bar_labels(self, y_order_ax1: List[str], y_order_ax2: List[str]) -> None:
        add_text_labels_to_bars(
            df=self._mwpr_df_org_name,
            ax=self._bar_plot_grid.ax1,
            y_column=PREFERRED_AFFILIATION_NAME_COLUMN,
            y_order=y_order_ax1,
        )
        add_text_labels_to_bars(
            df=self._mwpr_filtered_df_org_name,
            ax=self._bar_plot_grid.ax2,
            y_column=PREFERRED_AFFILIATION_NAME_COLUMN,
            y_order=y_order_ax2,
        )

    def _plot_bars_onto_grid(self, y_order_ax1: List[str], y_order_ax2: List[str]) -> None:
        plot_bar(ax=self._bar_plot_grid.ax1, data=self._mwpr_df_org_name, y_order=y_order_ax1)
        plot_bar(
            ax=self._bar_plot_grid.ax2, data=self._mwpr_filtered_df_org_name, y_order=y_order_ax2
        )

    def _compute_org_name_mwprs(self) -> None:
        self._mwpr_df_org_name = compute_mwpr_for_affiliation_class(
            node_df=self._node_df,
            n_groups=self.n_orgs,
            group_column=PREFERRED_AFFILIATION_NAME_COLUMN,
            min_samples=self.min_samples_org_name,
        ).to_concatenated()

        self._mwpr_filtered_df_org_name = compute_mwpr_for_affiliation_class(
            node_df=self._filtered_node_df,
            n_groups=self.n_orgs,
            group_column=PREFERRED_AFFILIATION_NAME_COLUMN,
            min_samples=self.min_samples_org_name,
        ).to_concatenated()

    def _compute_org_type_mwprs(self) -> None:
        self._mwpr_df_org_type = compute_mwpr_for_affiliation_class(
            node_df=self._node_df,
            n_groups=_N_GROUPS_ORG_TYPE,
            group_column=ORGANISATION_TYPE_COLUMN,
            min_samples=0,
        )

        self._mwpr_filtered_df_org_type = compute_mwpr_for_affiliation_class(
            node_df=self._filtered_node_df,
            n_groups=_N_GROUPS_ORG_TYPE,
            group_column=ORGANISATION_TYPE_COLUMN,
            min_samples=0,
        )
