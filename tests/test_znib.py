import itertools
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from src.mma.constants import (
    AFFILIATION_ID_FROM_COLUMN,
    AFFILIATION_ID_TO_COLUMN,
    DURATION_S_COLUMN,
    FROM_NODE_COLUMN,
    ORGANISATION_TYPE_COLUMN,
    TO_NODE_COLUMN,
)
from src.mma.network.utils import create_graph_from_links
from src.mma.utils.utils import filter_organization_types
from src.mma.znib.configuration import ZINBConfig
from src.mma.znib.utils import enrich_edges_with_org_info, get_znib_edges
from src.mma.znib.znib import ZNIB


@pytest.mark.parametrize(
    "org_type_filter, expected_length_edge_df",
    [
        (["uni", "res"], 3),
        (["uni"], 1),
        (["res"], 2),
        (["comp"], 0),
    ],
)
def test_filter_organization_types(
    link_df: pd.DataFrame, org_type_filter: List[str], expected_length_edge_df: int
) -> None:
    # create edges
    edge_graph = create_graph_from_links(link_gdf=link_df)
    edges = edge_graph.gdf

    # filter edges
    filtered_edges = filter_organization_types(df=edges, org_types=org_type_filter)
    from_org_types = filtered_edges[ORGANISATION_TYPE_COLUMN].unique()
    to_org_types = filtered_edges[f"{ORGANISATION_TYPE_COLUMN}_to"].unique()

    # org types must be in given org type filter list
    assert all(item in org_type_filter for item in from_org_types)
    assert all(item in org_type_filter for item in to_org_types)

    # filtered edges must be excepted length
    assert len(filtered_edges) == expected_length_edge_df


def test_get_znib_edges(link_df: pd.DataFrame) -> None:
    # create edges
    edge_graph = create_graph_from_links(link_gdf=link_df)
    edges = edge_graph.gdf

    # create all possible combinations from unique from node to node ids
    znib_edges = get_znib_edges(edge_gdf=edges)

    # check all possible combinations
    nodes = edges[FROM_NODE_COLUMN].tolist() + edges[TO_NODE_COLUMN].tolist()
    pairs = list(itertools.combinations(np.unique(np.asarray(nodes)), 2))
    assert len(znib_edges) == len(pairs)

    for pair in pairs:
        pair_subset_df = znib_edges[
            (znib_edges[FROM_NODE_COLUMN] == pair[0]) & (znib_edges[TO_NODE_COLUMN] == pair[1])
        ]
        assert len(pair_subset_df) == 1


def test_enrich_edges_with_org_info(link_df: pd.DataFrame, route_df: pd.DataFrame) -> None:
    # create edges
    edge_graph = create_graph_from_links(link_gdf=link_df)
    edges = edge_graph.gdf

    enriched_edges = enrich_edges_with_org_info(edge_gdf=edges, route_df=route_df)

    # check merged travel times
    assert np.all(
        enriched_edges[[FROM_NODE_COLUMN, TO_NODE_COLUMN]].values
        == route_df[[AFFILIATION_ID_FROM_COLUMN, AFFILIATION_ID_TO_COLUMN]].values
    )
    assert np.all(enriched_edges[DURATION_S_COLUMN] == route_df[DURATION_S_COLUMN])

    # same_org
    same_org_df = enriched_edges[
        enriched_edges[ORGANISATION_TYPE_COLUMN] == enriched_edges[f"{ORGANISATION_TYPE_COLUMN}_to"]
    ]
    assert np.all(same_org_df.same_org)

    # uni_uni
    same_org_df = enriched_edges[
        (enriched_edges[ORGANISATION_TYPE_COLUMN] == "uni")
        & (enriched_edges[f"{ORGANISATION_TYPE_COLUMN}_to"] == "uni")
    ]
    assert np.all(same_org_df.uni_uni)

    # res_res
    same_org_df = enriched_edges[
        (enriched_edges[ORGANISATION_TYPE_COLUMN] == "res")
        & (enriched_edges[f"{ORGANISATION_TYPE_COLUMN}_to"] == "res")
    ]
    assert np.all(same_org_df.res_res)

    # res_uni
    same_org_df = enriched_edges[
        (
            (enriched_edges[ORGANISATION_TYPE_COLUMN] == "res")
            & (enriched_edges[f"{ORGANISATION_TYPE_COLUMN}_to"] == "uni")
        )
        | (
            (enriched_edges[ORGANISATION_TYPE_COLUMN] == "uni")
            & (enriched_edges[f"{ORGANISATION_TYPE_COLUMN}_to"] == "res")
        )
    ]
    assert np.all(same_org_df.res_uni)


def test_znib_pipline(
    article_df: pd.DataFrame, affiliation_gdf: gpd.GeoDataFrame, route_df: pd.DataFrame
) -> None:
    znib = ZNIB(article_df=article_df, affiliation_gdf=affiliation_gdf)

    model_data = znib.enrich_edges_with_org_info(route_df=route_df)

    # check all possible combinations
    nodes = znib.edge.gdf[FROM_NODE_COLUMN].tolist() + znib.edge.gdf[TO_NODE_COLUMN].tolist()
    pairs = list(itertools.combinations(np.unique(np.asarray(nodes)), 2))
    assert len(model_data) == len(pairs)

    test_config = ZINBConfig(
        dependent_var="affiliation_edge_count",
        predictor_vars=[
            "ln_prod_article_count",
            "ln_duration",
            "uni_uni",
            "res_res",
        ],
        inflation_var="ln_prod_article_count",
        max_iterations=1000,
        alpha_reg=3.0,
    )
    znib.fit_znib(config=test_config, enriched_edge_df=model_data)
