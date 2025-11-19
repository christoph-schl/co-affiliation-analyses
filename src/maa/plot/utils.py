from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from maa.constants.constants import (
    AFFILIATION_CLASS_COLUMN,
    COVER_DATE_COLUMN,
    HAZEN_PERCENTILE_COLUMN,
    MWPR_COLUMN,
    ORG_TYPE_COLORS,
    ORGANISATION_TYPE_COLUMN,
    PREFERRED_AFFILIATION_NAME_COLUMN,
)
from maa.impact.utils import AffiliationMwpr, AffiliationType

_N_ROWS = 2
_N_COLS = 2

_COLOR_PALETTE: Dict[str, str] = {
    AffiliationType.AA.value: "#696969",  # grey
    AffiliationType.FA.value: "#1f77b4",  # blue
    AffiliationType.LA.value: "#ff7f0e",  # orange
}

# Constants for marker styling
_MARKER_STYLE = {
    "color": "k",
    "facecolors": "none",
    "edgecolors": "black",
    "linewidths": 1,
    "marker": "D",
    "s": 50,
    "zorder": 3,
}

_OFFSET_MAP = {
    "all": 0.0,
    "first": -0.2,
    "last": 0.2,
}


@dataclass
class Grid:
    fig: plt.Figure
    ax1: plt.Axes
    ax2: plt.Axes


def get_plotting_grid(
    create_hline: bool = True,
    title_a: Optional[str] = "(a) All affiliation links",
    title_b: Optional[str] = "(b) Filtered links (≥2 links by authors, ≥2 yrs apart)",
    y_label: str = "Hazen percentile [%]",
    x_label: str = "Organisation",
    deactivate_ax2_ylabels: bool = True,
    figure_size: Tuple[int, int] = (12, 5),
) -> Grid:
    """
    Create a Matplotlib figure with two vertically stacked subplots and a shared label area.

    :param create_hline:
        If True, draw a horizontal dashed line at 50; otherwise draw a vertical one.
    :param title_a:
        Title for the first subplot (top).
    :param title_b:
        Title for the second subplot (bottom).
    :param y_label:
        Label for the y-axis of the first subplot.
    :param x_label:
        Label for the x-axis (shown above the shared legend area).
    :param deactivate_ax2_ylabels:
        Whether to hide y-axis labels and ticks on the second subplot.
    :param figure_size:
        Tuple specifying the figure size in inches (width, height).
    :return:
        Tuple containing the Matplotlib figure and the two subplot axes (fig, ax1, ax2).
    """

    fig = plt.figure(figsize=figure_size, layout="constrained")
    gs = GridSpec(nrows=_N_ROWS, ncols=_N_COLS, figure=fig, height_ratios=[1, 0.1])
    gs.update(wspace=0.0, hspace=0.0)

    # Main subplots
    ax1 = fig.add_subplot(gs[0:1])
    ax2 = fig.add_subplot(gs[1:2])

    # Titles
    ax1.set_title(title_a, fontsize=13, pad=10, loc="left")
    ax2.set_title(title_b, fontsize=13, pad=10, loc="left")

    # Legend / shared label axis
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.set_xlabel(x_label, labelpad=0, fontsize=14)
    legend_ax.xaxis.set_label_position("top")
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)

    # Y-labels
    ax1.set_ylabel(y_label, fontsize=12)
    if deactivate_ax2_ylabels:
        ax2.set_ylabel(None)
        ax2.set_yticklabels([])

    # Reference lines
    line_func = ax1.axhline if create_hline else ax1.axvline
    line_func(50, color="k", linewidth=1.0, linestyle="--")
    (ax2.axhline if create_hline else ax2.axvline)(50, color="k", linewidth=1.0, linestyle="--")

    grid = Grid(fig=fig, ax1=ax1, ax2=ax2)

    return grid


def add_text_labels_to_bars(
    df: pd.DataFrame,
    ax: plt.Axes,
    *,
    y_order: Sequence[str],
    y_column: str,
    hue_column: str = "affiliation_class",
    samples_column: str = "samples",
    fontsize: int = 8,
    offset_frac: float = 0.01,
    label_fmt: str = "n: {}",
) -> None:
    """
    Annotate horizontal bar patches in `ax` with sample counts from `df`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the sample counts and the category columns.
    ax : matplotlib.axes.Axes
        Axes containing the horizontal bar plot (patches).
    hue_order : Sequence[str]
        Ordered sequence of hue/category labels used when plotting (left-to-right per y).
    y_order : Sequence[str]
        Ordered sequence of y categories (top-to-bottom ordering used when plotting).
    y_column : str
        Column name in `df` that contains the y category.
    hue_column : str, optional
        Column name in `df` that contains hue values (default "affiliation_class").
    samples_column : str, optional
        Column name in `df` with the sample counts (default "samples").
    fontsize : int, optional
        Font size for the annotations.
    offset_frac : float, optional
        Fraction of the x-axis width to offset the label to the right of the bar
        (default 0.01 = 1%).
    label_fmt : str, optional
        Format string for the label; the sample value is inserted with `str.format`.
        Default: "n: {}".
    """
    # Validate minimal assumptions
    if y_column not in df.columns:
        raise KeyError(f"y_column '{y_column}' not in DataFrame")
    if hue_column not in df.columns:
        raise KeyError(f"hue_column '{hue_column}' not in DataFrame")
    if samples_column not in df.columns:
        # allow missing samples column but warn and use empty strings
        samples_map = {}
    else:
        # Map (y, hue) -> sample (kept as string for clean display)
        samples_map = (
            df.set_index([y_column, hue_column])[samples_column].dropna().astype(str).to_dict()
        )

    # numeric tick positions for the y-categories
    yticks = np.asarray(ax.get_yticks(), dtype=float)

    # how many bars have been seen per y-category (to infer hue index)
    counts_per_y: Dict[str, int] = defaultdict(int)

    # offset in data coordinates (fraction of x-range)
    x0, x1 = ax.get_xlim()
    x_offset = (x1 - x0) * float(offset_frac)

    # annotate each horizontal bar patch
    for rect in ax.patches:
        # center y position of the bar
        y_center = rect.get_y() + rect.get_height() / 2.0

        # find nearest y category index (safeguarded)
        if yticks.size == 0:
            # nothing to match against
            continue
        idx = int(np.argmin(np.abs(yticks - y_center)))
        if idx < 0 or idx >= len(y_order):
            # tick matched outside of provided y_order; skip
            continue
        y_cat = y_order[idx]

        # determine hue index for this y (0 => first hue in hue_order)
        hue_idx = counts_per_y[y_cat]
        counts_per_y[y_cat] += 1  # increment now to keep in sync even if skipped

        hue_order = [aff.value for aff in AffiliationType]
        if hue_idx >= len(hue_order):
            # more bars present than expected — skip gracefully
            continue
        hue_label = hue_order[hue_idx]

        # lookup sample value, default to empty string if not found
        sample = samples_map.get((y_cat, hue_label), "")
        label = label_fmt.format(sample)

        # x position is the width of the bar (horizontal bar); place label just to the right
        width = rect.get_width()
        ax.text(
            width + x_offset,
            y_center,
            label,
            va="center",
            ha="left",
            fontsize=fontsize,
            color="black",
        )


def add_legend(
    ax: plt.Axes,
    fig: plt.Figure,
    *,
    add_all_and_mwpr: bool = True,
    placement: str = "lower center",
    n_legend_columns: int = 3,
    bbox_to_anchor: Tuple[float, float] = (0.5, -0.01),
    fontsize: int = 13,
    override_handles: Optional[Mapping[str, Any]] = None,
    preferred_order: Sequence[str] = ("resi", "univ", "gov", "comp", "hosp", "coll", "ngo"),
) -> None:
    """
    Build and attach a tidy legend to `fig` for the plotted items in `ax`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes that contain the plotted artists (used to extract default handles/labels).
    fig : matplotlib.figure.Figure
        Figure to which the legend will be attached (using `fig.legend()`).
    add_all_and_mwpr : bool, optional
        If True, insert the special "All" patch and an mwPR marker into the legend.
    placement : str, optional
        The location string passed to `Figure.legend(..., loc=placement)`.
    n_legend_columns : int, optional
        Number of columns in the legend.
    bbox_to_anchor : tuple, optional
        bounding box anchor for the legend (x, y).
    fontsize : int, optional
        Font size for legend text.
    override_handles : Mapping[str, Any], optional
        Optional mapping from label -> artist to override the automatically-detected handle
        for a given label. Only keys that appear in the axes' labels are applied.
    preferred_order : Sequence[str], optional
        If any label in `preferred_order` appears among the detected labels, the legend
        will be reordered to follow this sequence (missing labels are skipped).

    Returns
    -------
    matplotlib.legend.Legend
        The created legend instance attached to `fig`.
    """
    # grab existing handles/labels from the axes (preserves plotting order)
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle: Dict[str, Any] = dict(zip(labels, handles))

    # apply any user-supplied overrides (but only for labels that exist)
    if override_handles:
        for label, custom_handle in override_handles.items():
            if label in label_to_handle:
                label_to_handle[label] = custom_handle

    # Optionally reorder labels if preferred_order items are present
    final_labels = list(label_to_handle.keys())
    if any(lbl in final_labels for lbl in preferred_order):
        # keep the preferred_order but only include labels that exist
        ordered = [lbl for lbl in preferred_order if lbl in label_to_handle]
        # append any remaining labels that weren't in preferred_order, preserving their original
        # order
        remaining = [lbl for lbl in final_labels if lbl not in ordered]
        final_labels = ordered + remaining

    final_handles = [label_to_handle[lbl] for lbl in final_labels]

    # optionally, add "All" patch and mwPR marker
    if add_all_and_mwpr:
        # Prepare special legend artists
        mwpr_marker = Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            color="black",
            markerfacecolor="none",
            markersize=7,
        )
        all_patch = Patch(facecolor="none", edgecolor="darkgrey", linewidth=1.5, label="All")

        # Decide safe insertion points: front (first), back (last). If none exist, just create a
        # small list.
        if final_handles:
            first_handle, last_handle = final_handles[0], final_handles[-1]
            # build new lists as [first, All, last, mwPR] while avoiding duplication if lists
            # are small. if there's only one handle, use it for both first and last positions.
            new_handles = [first_handle, all_patch]
            new_labels = [final_labels[0], f"{AffiliationType.AA.value}"]

            if len(final_handles) > 1:
                new_handles.append(last_handle)
                new_labels.append(final_labels[-1])
            else:
                # only one existing handle — repeat it for the "last" slot so structure remains
                # similar
                new_handles.append(first_handle)
                new_labels.append(final_labels[0])

            new_handles.append(mwpr_marker)
            new_labels.append(
                f"mwPR [%] ({AffiliationType.FA.value}, "
                f"{AffiliationType.AA.value}, {AffiliationType.LA.value})"
            )

            final_handles = new_handles
            final_labels = new_labels
        else:
            # no existing handles — just show All and mwPR
            final_handles = [all_patch, mwpr_marker]
            final_labels = [f"{AffiliationType.AA.value}", "mwPR [%]"]

    # finally, create and return the figure legend
    fig.legend(
        final_handles,
        final_labels,
        loc=placement,
        ncol=n_legend_columns,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        fontsize=fontsize,
    )


def plot_bar(ax: plt.Axes, data: pd.DataFrame, y_order: List[str]) -> None:
    """Single-axis barplot wrapper to keep plotting consistent."""
    sns.barplot(
        y=PREFERRED_AFFILIATION_NAME_COLUMN,
        x=MWPR_COLUMN,
        hue=AFFILIATION_CLASS_COLUMN,
        data=data,
        palette=_COLOR_PALETTE,
        order=y_order,
        hue_order=[a.value for a in AffiliationType],
        ax=ax,
    )
    # tidy axis style
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right")

    # remove per-axis legend; we'll add a single figure legend later
    if getattr(ax, "legend_", None):
        ax.legend_.remove()


def add_mwpr_marker(
    axes: plt.Axes, mwpr: pd.DataFrame, mwpr_type: str, org_types: List[str]
) -> None:
    """
    Add markers to the violine plots to highlight the mwPR
    (first, all, last) for each organization type.

    :param axes: The matplotlib Axes to draw the markers on.
    :param mwpr: The DataFrame with mwPR values.
    :param mwpr_type: Type of mwPR marker to plot ('all', 'first', 'last').
    :param org_types: Ordered list of organization types corresponding to x-axis positions.
    :return: None
    """
    offset = _OFFSET_MAP.get(mwpr_type)
    if offset is None:
        raise ValueError(
            f"Invalid mwpr_type '{mwpr_type}'. " f"Expected one of {list(_OFFSET_MAP.keys())}."
        )

    mwpr_dict = mwpr.set_index(ORGANISATION_TYPE_COLUMN)[MWPR_COLUMN].to_dict()
    for position, org_type in enumerate(org_types):
        mwpr_value = mwpr_dict.get(org_type)
        if mwpr_value is None:
            continue

        axes.scatter(position + offset, mwpr_value, **_MARKER_STYLE)


def plot_violine(
    ax: plt.Axes, nodes_df: pd.DataFrame, mwpr: AffiliationMwpr, org_type_order: List[str]
) -> None:
    """
    Plot violin distributions of Hazen percentiles for each organisation type and
    annotate the plot with mwPR markers for 'all', 'first' and 'last' groups.

    :param ax: Matplotlib Axes to draw the plot on.
    :param nodes_df: DataFrame containing the raw node records (must include columns
                     referenced by `org_type` and `hazen_perc_med` columns).
    :param mwpr: AffiliationMwpr dataclass containing `all`, `first`, and `last` DataFrames
                 (each mapping organisation types to mwPR values).
    :param org_type_order: Ordered list of organisation types controlling x-axis order.
    """

    # 1) draw split/hue violins per affiliation class (colored, split)
    sns.violinplot(
        data=nodes_df,
        x=ORGANISATION_TYPE_COLUMN,
        y=HAZEN_PERCENTILE_COLUMN,
        hue=AFFILIATION_CLASS_COLUMN,
        split=True,
        inner="quart",
        order=org_type_order,
        width=1,
        cut=0,
        density_norm="width",
        saturation=1,
        ax=ax,
        bw_adjust=0.8,
        gap=0.2,
    )

    # 2) overlay a neutral outline violin for emphasis (no fill)
    sns.violinplot(
        data=nodes_df,
        x=ORGANISATION_TYPE_COLUMN,
        y=HAZEN_PERCENTILE_COLUMN,
        order=org_type_order,
        inner="quart",
        color="darkgrey",
        width=1,
        cut=0,
        density_norm="width",
        ax=ax,
        bw_adjust=0.8,
        saturation=1,
        linewidth=2,
        fill=False,
    )

    for mwpr_type in ("all", "first", "last"):
        add_mwpr_marker(
            axes=ax,
            mwpr=getattr(mwpr, mwpr_type),
            mwpr_type=mwpr_type,
            org_types=org_type_order,
        )

    if ax.legend_ is not None:
        ax.legend_.remove()


def _to_matplotlib_color(col: Any) -> Any:
    """
    Convert a color value into something matplotlib accepts.

    Handles:
    - objects with .red/.green/.blue in 0-255 range,
    - already-valid matplotlib color strings or RGB tuples,
    - anything else is returned unchanged (matplotlib will error if invalid).
    """
    # color-like object with .red/.green/.blue attributes (0-255)
    if hasattr(col, "red") and hasattr(col, "green") and hasattr(col, "blue"):
        return col.red / 255.0, col.green / 255.0, col.blue / 255.0
    # tuple/list of ints 0-255
    if (
        isinstance(col, (tuple, list))
        and len(col) == 3
        and all(isinstance(v, (int, float)) for v in col)
    ):
        # if ints in 0-255 convert to 0-1
        if max(col) > 1:
            return tuple(v / 255.0 for v in col)
        return tuple(col)

    # otherwise assume matplotlib can handle it (named color, hex, rgb in 0-1, etc.)
    return col


def plot_time_series(
    df: pd.DataFrame,
    group_column: str = ORGANISATION_TYPE_COLUMN,
    axes: Optional[plt.Axes] = None,
    *,
    figsize: Tuple[int, int] = (10, 6),
    linewidth: float = 3.5,
    marker: str = "o",
    marker_size: int = 7,
    legend_ncol_default: int = 2,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot mwPR time series pivoted by group column.
    :param df: DataFrame that must contain COVER_DATE_COLUMN, group_column and MWPR_COLUMN.
    :param group_column: Column used to pivot the data into separate time series.
    :param axes: Optional axis to draw on. If None, a new figure/axis is created.
    :param figsize: Figure size used when creating a new figure.
    :param linewidth: Line width for series.
    :param marker: Marker style for points.
    :param marker_size: Size of the markers.
    :param legend_ncol_default: Default number of legend columns when group_column != 'org_type'.
    :return: The matplotlib Figure and Axes used.
    """

    df = df.copy()

    # Pivot into time-indexed series matrix
    if COVER_DATE_COLUMN not in df.columns:
        raise ValueError(f"DataFrame must contain column '{COVER_DATE_COLUMN}'")
    if MWPR_COLUMN not in df.columns:
        raise ValueError(f"DataFrame must contain column '{MWPR_COLUMN}'")

    # ensure datetime index / column
    df[COVER_DATE_COLUMN] = pd.to_datetime(df[COVER_DATE_COLUMN])
    pivot = df.pivot(index=COVER_DATE_COLUMN, columns=group_column, values=MWPR_COLUMN).sort_index()

    # Build color map for org types (convert to matplotlib-friendly colors)
    org_type_colors_rgb: Dict[str, Any] = (
        {k: _to_matplotlib_color(v) for k, v in ORG_TYPE_COLORS.items()}
        if group_column == ORGANISATION_TYPE_COLUMN and "ORG_TYPE_COLORS" in globals()
        else {}
    )

    # Create (or accept) axes
    if axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = axes
        fig = ax.get_figure()

    x = pivot.index

    # Plot each column with consistent styling
    for col in pivot.columns:
        y = pivot[col].values
        # Skip entirely NaN series
        if pd.isna(y).all():
            continue

        # Choose color only if we have a mapping and we're plotting org_type
        color = None
        if group_column == ORGANISATION_TYPE_COLUMN:
            color = org_type_colors_rgb.get(col, "gray")

        ax.plot(
            x,
            y,
            linewidth=linewidth,
            marker=marker,
            markersize=marker_size,
            label=str(col),
            color=color,
            solid_capstyle="round",
        )

    # Reference line at 50
    ax.axhline(50, color="grey", linewidth=2.0, linestyle="--", zorder=0)

    # Styling
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("")  # keep caller free to set if desired

    # Legend layout (tuned for org_type vs other groupings)
    ncol = 7 if group_column == ORGANISATION_TYPE_COLUMN else legend_ncol_default
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=ncol,
        frameon=False,
    )

    # If we created the figure, make room for the legend
    if axes is None:
        fig.subplots_adjust(bottom=0.22)

    if ax.legend_ is not None:
        ax.legend_.remove()

    return fig, ax
