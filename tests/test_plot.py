import pandas as pd

from src.mma.constants import (
    MWPR_COLUMN,
    ORGANISATION_TYPE_COLUMN,
    PREFERRED_AFFILIATION_NAME_COLUMN,
)
from src.mma.impact.utils import compute_mwpr_for_affiliation_class
from src.mma.network.utils import retain_affiliation_links_with_min_year_gap
from src.mma.plot.plot import ImpactPlot

_NUMBER_OF_GROUPS = 10


def test_compute_mwpr_for_affiliation_class(link_df: pd.DataFrame, impact_df: pd.DataFrame) -> None:

    filtered_link_df = retain_affiliation_links_with_min_year_gap(link_gdf=link_df, min_year_gap=2)
    plot = ImpactPlot(link_gdf=link_df, filtered_link_gdf=filtered_link_df, impact_df=impact_df)

    _test_mwpr_unfiltered_preferred_name(impact_plot=plot)
    _test_mwpr_filtered_preferred_name(impact_plot=plot)

    _test_mwpr_unfiltered_org_type(impact_plot=plot)
    _test_mwpr_filtered_org_type(impact_plot=plot)


def _test_mwpr_unfiltered_preferred_name(impact_plot: ImpactPlot) -> None:
    mwpr = compute_mwpr_for_affiliation_class(
        node_df=impact_plot.node_df, n_groups=_NUMBER_OF_GROUPS
    )
    assert mwpr.all[MWPR_COLUMN].round(2).tolist() == [99.76, 89.88, 86.41, 86.41, 80.0]
    assert mwpr.all[PREFERRED_AFFILIATION_NAME_COLUMN].tolist() == [
        "Resi_11",
        "Resi_10",
        "Univ_8",
        "Univ_9",
        "Resi_1",
    ]

    assert mwpr.first[MWPR_COLUMN].round(2).tolist() == [89.88, 86.41]
    assert mwpr.first[PREFERRED_AFFILIATION_NAME_COLUMN].tolist() == ["Resi_10", "Univ_8"]

    assert mwpr.last[MWPR_COLUMN].round(2).tolist() == [99.76, 86.41, 80.0]
    assert mwpr.last[PREFERRED_AFFILIATION_NAME_COLUMN].tolist() == ["Resi_11", "Univ_9", "Resi_1"]


def _test_mwpr_filtered_preferred_name(impact_plot: ImpactPlot) -> None:
    mwpr_filtered = compute_mwpr_for_affiliation_class(
        node_df=impact_plot.filtered_node_df, n_groups=_NUMBER_OF_GROUPS
    )
    assert mwpr_filtered.all[MWPR_COLUMN].round(2).tolist() == [86.41, 86.41]
    assert mwpr_filtered.all[PREFERRED_AFFILIATION_NAME_COLUMN].tolist() == ["Univ_8", "Univ_9"]

    assert mwpr_filtered.first[MWPR_COLUMN].round(2).tolist() == [86.41]
    assert mwpr_filtered.first[PREFERRED_AFFILIATION_NAME_COLUMN].tolist() == ["Univ_8"]

    assert mwpr_filtered.last[MWPR_COLUMN].round(2).tolist() == [86.41]
    assert mwpr_filtered.last[PREFERRED_AFFILIATION_NAME_COLUMN].tolist() == ["Univ_9"]


def _test_mwpr_unfiltered_org_type(impact_plot: ImpactPlot) -> None:
    mwpr = compute_mwpr_for_affiliation_class(
        node_df=impact_plot.node_df,
        n_groups=_NUMBER_OF_GROUPS,
        group_column=ORGANISATION_TYPE_COLUMN,
    )
    assert mwpr.all[MWPR_COLUMN].round(2).tolist() == [89.88, 86.41]
    assert mwpr.all[ORGANISATION_TYPE_COLUMN].tolist() == ["res", "uni"]

    assert mwpr.first[MWPR_COLUMN].round(2).tolist() == [89.88, 86.41]
    assert mwpr.first[ORGANISATION_TYPE_COLUMN].tolist() == ["res", "uni"]

    assert mwpr.last[MWPR_COLUMN].round(2).tolist() == [89.88, 86.41]
    assert mwpr.last[ORGANISATION_TYPE_COLUMN].tolist() == ["res", "uni"]


def _test_mwpr_filtered_org_type(impact_plot: ImpactPlot) -> None:
    mwpr_filtered = compute_mwpr_for_affiliation_class(
        node_df=impact_plot.filtered_node_df,
        n_groups=_NUMBER_OF_GROUPS,
        group_column=ORGANISATION_TYPE_COLUMN,
    )

    for mwpr in [mwpr_filtered.all, mwpr_filtered.first, mwpr_filtered.last]:
        assert mwpr[MWPR_COLUMN].round(2).tolist() == [86.41]
        assert mwpr[ORGANISATION_TYPE_COLUMN].tolist() == ["uni"]
