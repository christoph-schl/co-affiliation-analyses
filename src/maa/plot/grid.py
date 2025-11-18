from dataclasses import dataclass
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.maa.plot.configuration import (
    PLOT_CONFIGS,
    FigureConfig,
    PlotLegendConfig,
)
from src.maa.plot.utils import add_legend


@dataclass
class PlotGrid:
    """Encapsulates a figure with two stacked axes and common formatting."""

    fig: plt.Figure
    ax1: plt.Axes
    ax2: plt.Axes
    legend_ax: plt.Axes

    @classmethod
    def from_config(cls, config: FigureConfig) -> "PlotGrid":
        """Builds a PlotGrid from a FigureConfig."""
        fig = plt.figure(figsize=config.figure_size, layout="constrained")
        gs = GridSpec(nrows=2, ncols=2, figure=fig, height_ratios=config.height_ratio)
        gs.update(wspace=0.0, hspace=0.0)

        # main axes
        ax1 = fig.add_subplot(gs[0:1])
        ax2 = fig.add_subplot(gs[1:2])

        # titels
        ax1.set_title(config.title_a, fontsize=16, pad=10, loc="left")
        ax2.set_title(config.title_b, fontsize=16, pad=10, loc="left")

        # shared label (legend area)
        legend_ax = fig.add_subplot(gs[1, :])
        legend_ax.set_xlabel(config.x_label, labelpad=0, fontsize=16)
        legend_ax.xaxis.set_label_position("top")
        legend_ax.set_xticks([])
        legend_ax.set_yticks([])

        for spine in legend_ax.spines.values():
            spine.set_visible(False)

        # Y-labels
        ax1.set_ylabel(config.y_label, labelpad=0, fontsize=16)
        if config.deactivate_ax2_ylabels:
            ax2.set_ylabel(None)
            ax2.set_yticklabels([])

        # Reference lines
        line_func = ax1.axhline if config.create_hline else ax1.axvline
        line_func(50, color="k", linewidth=1.0, linestyle="--")
        (ax2.axhline if config.create_hline else ax2.axvline)(
            50, color="k", linewidth=1.0, linestyle="--"
        )

        for ax in [ax1, ax2]:
            ax.tick_params(axis="x", labelsize=config.tickx_label_size)
            ax.tick_params(axis="y", labelsize=config.ticky_label_size)

            if config.ylim_bottom is not None:
                ax.set_ylim(bottom=config.ylim_bottom)

            if config.ylim_top is not None:
                ax.set_ylim(top=config.ylim_top)

        return cls(fig=fig, ax1=ax1, ax2=ax2, legend_ax=legend_ax)

    def add_legends_from_plot_config(
        self,
        config: PlotLegendConfig,
        label_handle: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Apply legends to ax1 / ax2 as described by the plot config.
        Uses a graceful fallback if the named config is not found.
        """
        # Accept either full PLOT_CONFIGS (PlotConfig) or legacy LEGEND_BY_PLOT.

        # pair up the config with axes explicitly
        pairs = [
            (config.axes1, self.ax1),
            (config.axes2, self.ax2),
        ]

        for cfg, ax in pairs:
            if cfg is None:
                continue
            add_legend(
                ax=ax,
                fig=self.fig,
                add_all_and_mwpr=cfg.add_all_and_mwpr,
                n_legend_columns=cfg.n_legend_columns,
                placement=cfg.placement,
                bbox_to_anchor=cfg.bbox_to_anchor,
                fontsize=cfg.fontsize,
                override_handles=label_handle,
            )

    @classmethod
    def from_plot_name(cls, plot_name: str) -> "PlotGrid":
        """Creates a PlotGrid directly from your PLOT_CONFIGS by plot key."""
        cfg = PLOT_CONFIGS.get(plot_name)
        if not cfg:
            raise ValueError(f"No plot configuration found for '{plot_name}'")
        return cls.from_config(cfg.figure)

    def save(self, path: str, **kwargs: Any) -> None:
        """Convenience helper to save the figure."""
        self.fig.savefig(path, bbox_inches="tight", **kwargs)
