# configuration.py
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


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
    title_a: Optional[str] = "(a) All affiliation links"
    title_b: Optional[str] = "(b) Filtered links (≥2 links by authors, ≥2 yrs apart)"
    y_label: str = "Hazen percentile [%]"
    x_label: str = "Organisation"
    deactivate_ax2_ylabels: bool = True
    figure_size: Tuple[int, int] = (12, 5)


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
                bbox_to_anchor=(0.22, 0.00),
                n_legend_columns=4,
                fontsize=13,
            ),
            axes2=None,
        ),
        figure=FigureConfig(
            # override any figure-level defaults if you want
            create_hline=True,
            title_a="(a) All affiliation links",
            title_b="(b) Filtered links (≥2 links by authors, ≥2 yrs apart)",
            y_label="mwPR [%]",
            x_label="Publication year",
            deactivate_ax2_ylabels=False,
            figure_size=(12, 6),
        ),
    ),
    "timeseries_by_preferred_name": PlotConfig(
        legend=PlotLegendConfig(
            axes1=LegendConfig(
                placement="lower center",
                bbox_to_anchor=(0.25, 0.18),
                n_legend_columns=2,
                fontsize=13,
            ),
            axes2=LegendConfig(
                placement="lower center",
                bbox_to_anchor=(0.746, 0.18),
                n_legend_columns=2,
                fontsize=13,
            ),
        ),
    ),
    "violine_by_org_type": PlotConfig(
        legend=PlotLegendConfig(
            axes1=LegendConfig(
                placement="lower center",
                bbox_to_anchor=(0.5, -0.01),
                n_legend_columns=4,
                fontsize=13,
                add_all_and_mwpr=True,
            ),
            axes2=None,
        ),
        figure=FigureConfig(
            create_hline=True,
            title_a="(a) All affiliation links",
            title_b="(b) Filtered links (≥2 links by authors, ≥2 yrs apart)",
            y_label="Hazen percentile [%]",
            x_label="Organisation",
            deactivate_ax2_ylabels=True,
            figure_size=(12, 5),
        ),
    ),
    "barplot_by_preferred_name": PlotConfig(
        legend=PlotLegendConfig(
            axes1=LegendConfig(
                placement="lower center",
                bbox_to_anchor=(0.25, 0.05),
                n_legend_columns=3,
                fontsize=13,
            ),
            axes2=None,
        ),
        figure=FigureConfig(
            # override any figure-level defaults if you want
            create_hline=False,
            title_a="(a) All affiliation links",
            title_b="(b) Filtered links (≥2 links by authors, ≥2 yrs apart)",
            y_label="",
            x_label="mwPR [%]",
            deactivate_ax2_ylabels=False,
            figure_size=(12, 6),
        ),
    ),
}

# Backwards compatibility helper (if you still want a plain LEGEND_BY_PLOT)
# This keeps old code working if it expects LEGEND_BY_PLOT variable.
LEGEND_BY_PLOT: Dict[str, PlotLegendConfig] = {
    name: cfg.legend for name, cfg in PLOT_CONFIGS.items()
}
