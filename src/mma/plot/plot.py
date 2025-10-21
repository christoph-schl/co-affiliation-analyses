from dataclasses import dataclass, field
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt

from src.mma.constants import (
    AFFILIATION_CLASS_COLUMN,
    MWPR_COLUMN,
    PREFERRED_AFFILIATION_NAME_COLUMN,
)
from src.mma.impact.impact import Impact
from src.mma.impact.utils import (
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
)
from src.mma.utils.utils import get_link_nodes

_COLOR_PALETTE: Dict[str, str] = {
    AffiliationType.ALL.value: "#696969",  # grey
    AffiliationType.FIRST.value: "#1f77b4",  # blue
    AffiliationType.LAST.value: "#ff7f0e",  # orange
}


def _get_y_order(mwpr: pd.DataFrame, x_column: str, y_column: str) -> List[str]:
    return (
        mwpr.loc[
            mwpr[AFFILIATION_CLASS_COLUMN] == AffiliationType.ALL.value, [y_column, x_column]
        ].sort_values(by=x_column, ascending=False)
    )[y_column].tolist()


@dataclass
class ImpactPlot(Impact):
    filtered_link_gdf: gpd.GeoDataFrame
    _bar_plot_grid: Grid = field(
        default_factory=lambda: get_plotting_grid(
            create_hline=False,
            deactivate_ax2_ylabels=False,
            figure_size=(12, 6),
            x_label="mwPR [%]",
        )
    )
    _filtered_node_df: Optional[pd.DataFrame] = field(default=None, init=False)
    _mwpr_df: Optional[pd.DataFrame] = field(default=None, init=False)
    _mwpr_filtered_df: Optional[pd.DataFrame] = field(default=None, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        self._filtered_node_df = get_link_nodes(link_gdf=self.filtered_link_gdf)
        self._filtered_node_df = merge_impact_measures_to_nodes(
            node_df=self._filtered_node_df, impact_df=self.impact_df
        )

        # apply affiliation aliases
        for df in [self._node_df, self._filtered_node_df]:
            df[PREFERRED_AFFILIATION_NAME_COLUMN] = df[PREFERRED_AFFILIATION_NAME_COLUMN].apply(
                lambda x: AFFILIATION_NAME_ALIASES.get(x, x)
            )

    def get_bar_plot(self, min_samples: int, n_groups: int) -> plt.Figure:
        """
        Generates a grouped bar plot showing mwPR values
        for different affiliation classes (all, first, or the last affiliation),
        using both the full and filtered node datasets.

        The grouping is performed using the `preferred_name` column, which identifies. The mwPR
        values are calculated separately for the full dataset and a filtered subset, allowing
        side-by-side comparison in the resulting bar plot.

        :param min_samples:
            Minimum number of samples required for a `preferred_name` group
            to be included in the mwPR calculation.
        :param n_groups:
            Number of groups into which the `preferred_name` column values
            will be divided during computation.
        :return:
            A matplotlib Figure object containing the completed bar plot visualization.
        """

        self._compute_mwprs(min_samples=min_samples, n_groups=n_groups)

        y_order_ax1 = _get_y_order(
            mwpr=self._mwpr_df, x_column=MWPR_COLUMN, y_column=PREFERRED_AFFILIATION_NAME_COLUMN
        )
        y_order_ax2 = _get_y_order(
            mwpr=self._mwpr_filtered_df,
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

    def _add_bar_labels(self, y_order_ax1: List[str], y_order_ax2: List[str]) -> None:
        add_text_labels_to_bars(
            df=self._mwpr_df,
            ax=self._bar_plot_grid.ax1,
            y_column=PREFERRED_AFFILIATION_NAME_COLUMN,
            y_order=y_order_ax1,
        )
        add_text_labels_to_bars(
            df=self._mwpr_filtered_df,
            ax=self._bar_plot_grid.ax2,
            y_column=PREFERRED_AFFILIATION_NAME_COLUMN,
            y_order=y_order_ax2,
        )

    def _plot_bars_onto_grid(self, y_order_ax1: List[str], y_order_ax2: List[str]) -> None:
        plot_bar(ax=self._bar_plot_grid.ax1, data=self._mwpr_df, y_order=y_order_ax1)
        plot_bar(ax=self._bar_plot_grid.ax2, data=self._mwpr_filtered_df, y_order=y_order_ax2)

    def _compute_mwprs(self, min_samples: int, n_groups: int) -> None:
        self._mwpr_df = compute_mwpr_for_affiliation_class(
            node_df=self._node_df,
            n_groups=n_groups,
            group_column=PREFERRED_AFFILIATION_NAME_COLUMN,
            min_samples=min_samples,
        )
        self._mwpr_filtered_df = compute_mwpr_for_affiliation_class(
            node_df=self._filtered_node_df,
            n_groups=n_groups,
            group_column=PREFERRED_AFFILIATION_NAME_COLUMN,
            min_samples=min_samples,
        )
