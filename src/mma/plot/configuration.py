# configuration.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_DEFAULT_TITLE_A = "(a) affiliation_links_all"
_DEFAULT_TITLE_B = "(b) affiliation_links_filtered"


# ---------- Legend dataclasses (unchanged, slightly renamed fields) ----------
@dataclass(frozen=True)
class LegendConfig:
    placement: str = "best"
    bbox_to_anchor: Tuple[float, float] = (0.5, -0.01)
    n_legend_columns: int = 1
    fontsize: int = 12
    add_all_and_mwpr: bool = False


@dataclass(frozen=True)
class PlotLegendConfig:
    axes1: Optional[LegendConfig]
    axes2: Optional[LegendConfig] = None


# Default legend parameters used when nothing else specified
DEFAULT_LEGEND = LegendConfig()


# ---------- New figure-level config dataclass ----------
@dataclass(frozen=True)
class FigureConfig:
    create_hline: bool = True
    title_a: Optional[str] = _DEFAULT_TITLE_A
    title_b: Optional[str] = _DEFAULT_TITLE_B
    y_label: str = "wPR [%]"
    x_label: str = "Organisation"
    deactivate_ax2_ylabels: bool = True
    figure_size: Tuple[int, int] = (12, 5)
    height_ratio: List[float] = field(default_factory=lambda: [1, 0.1])
    tickx_label_size: int = 14
    ticky_label_size: int = 14


# Default figure config
DEFAULT_FIGURE = FigureConfig()


# ---------- Combined per-plot config ----------
@dataclass(frozen=True)
class PlotConfig:
    legend: PlotLegendConfig
    figure: FigureConfig = DEFAULT_FIGURE


# ---------- Per-plot configs (examples) ----------
PLOT_CONFIGS: Dict[str, PlotConfig] = {
    "timeseries_by_org_type": PlotConfig(
        legend=PlotLegendConfig(
            axes1=LegendConfig(
                placement="lower center",
                bbox_to_anchor=(0.525, -0.009),
                n_legend_columns=9,
                fontsize=14,
            ),
            axes2=None,
        ),
        figure=FigureConfig(
            # override any figure-level defaults if you want
            create_hline=True,
            title_a=_DEFAULT_TITLE_A,
            title_b=_DEFAULT_TITLE_B,
            y_label="mwPR [%]",
            x_label="Publication year",
            deactivate_ax2_ylabels=True,
            figure_size=(12, 6),
            height_ratio=[1, 0.09],
        ),
    ),
    "timeseries_by_preferred_name": PlotConfig(
        legend=PlotLegendConfig(
            axes1=LegendConfig(
                placement="lower center",
                bbox_to_anchor=(0.525, -0.01),
                n_legend_columns=5,
                fontsize=13,
            ),
            axes2=None,
        ),
        figure=FigureConfig(
            create_hline=True,
            title_a=_DEFAULT_TITLE_A,
            title_b=_DEFAULT_TITLE_B,
            y_label="mwPR [%]",
            x_label="Publication year",
            deactivate_ax2_ylabels=True,
            figure_size=(12, 6),
            height_ratio=[1, 0.17],
        ),
    ),
    "violine_by_org_type": PlotConfig(
        legend=PlotLegendConfig(
            axes1=LegendConfig(
                placement="lower center",
                bbox_to_anchor=(0.5, -0.015),
                n_legend_columns=4,
                fontsize=13,
                add_all_and_mwpr=True,
            ),
            axes2=None,
        ),
        figure=FigureConfig(
            create_hline=True,
            title_a=_DEFAULT_TITLE_A,
            title_b=_DEFAULT_TITLE_B,
            y_label="wPR [%]",
            x_label="Organisation",
            deactivate_ax2_ylabels=True,
            figure_size=(12, 5),
            tickx_label_size=16,
            ticky_label_size=16,
        ),
    ),
    "barplot_by_preferred_name": PlotConfig(
        legend=PlotLegendConfig(
            axes1=LegendConfig(
                placement="lower center",
                bbox_to_anchor=(0.3, 0.00),
                n_legend_columns=3,
                fontsize=14,
            ),
            axes2=None,
        ),
        figure=FigureConfig(
            # override any figure-level defaults if you want
            create_hline=False,
            title_a=_DEFAULT_TITLE_A,
            title_b=_DEFAULT_TITLE_B,
            y_label="",
            x_label="mwPR [%]",
            deactivate_ax2_ylabels=False,
            figure_size=(12, 7),
            height_ratio=[1, 0.01],
        ),
    ),
}

# Backwards compatibility helper (if you still want a plain LEGEND_BY_PLOT)
# This keeps old code working if it expects LEGEND_BY_PLOT variable.
LEGEND_BY_PLOT: Dict[str, PlotLegendConfig] = {
    name: cfg.legend for name, cfg in PLOT_CONFIGS.items()
}
