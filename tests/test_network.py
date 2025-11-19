import math
from typing import List

import geopandas as gpd
import pandas as pd
import pytest

from maa.constants.constants import (
    AFFILIATION_EDGE_COUNT_COLUMN,
    ARTICLE_AUTHOR_ID_COLUMN,
    ARTICLE_COUNT_COLUMN,
    FROM_NODE_COLUMN,
)
from maa.network.network import (
    AffiliationNetworkProcessor,
    apply_parent_affiliation_id_and_idx,
)
from maa.network.utils import (
    create_affiliation_links,
    create_graph_from_links,
    retain_affiliation_links_with_min_year_gap,
)
from maa.utils.utils import get_affiliation_id_map, get_articles_per_author


def test_get_articles_per_author(article_df: pd.DataFrame) -> None:
    article_author_df = get_articles_per_author(article_df=article_df)
    assert len(article_author_df) == 5
    assert article_author_df.author_ids.tolist() == [
        "Author1",
        "Author1",
        "Author2",
        "Author1",
        "Author3",
    ]
    assert article_author_df.eid.tolist() == ["Eid1", "Eid2", "Eid3", "Eid4", "Eid4"]
    assert article_author_df.author_afids.tolist() == ["1-2-3", "1-2-3", "4-5", "4-5", "8"]


def test_get_affiliation_id_map(affiliation_gdf: pd.DataFrame) -> None:
    affiliation_map = get_affiliation_id_map(affiliation_gdf=affiliation_gdf)
    assert affiliation_map == {1: 8, 2: 9, 3: 8, 4: 10, 5: 11, 8: 8, 9: 9, 10: 10, 11: 11}


def test_apply_parent_affiliation_id_and_idx(
    article_df: pd.DataFrame, affiliation_gdf: gpd.GeoDataFrame
) -> None:
    article_author_df = get_articles_per_author(article_df=article_df)
    affiliation_map = get_affiliation_id_map(affiliation_gdf=affiliation_gdf)

    article_author_df = apply_parent_affiliation_id_and_idx(
        article_author_df=article_author_df, affiliation_map=affiliation_map
    )

    # "author_afids": ["1-2-3", "1-2-3", "4-5", "4-5;8"]
    # "affiliation_id": [1, 2, 3, 4, 5],
    # "affiliation_id_parent": [8, 9, 8, 10, 11]
    # the new affiliation id is the applied parent id
    assert article_author_df.author_afids.tolist() == [[8, 9], [8, 9], [10, 11], [10, 11], [8]]

    for row in article_author_df.itertuples():
        # test affiliation index
        assert row.affiliation_idx == {v: i for i, v in enumerate(row.author_afids)}


@pytest.mark.parametrize(
    "min_year_gap,expected_author_ids",
    [
        (0, ["Author1", "Author1", "Author2", "Author3"]),
        (1, ["Author1", "Author1"]),
        (2, ["Author1", "Author1"]),
        (3, []),
    ],
)
def test_retain_affiliation_links_with_min_year_gap(
    link_df: pd.DataFrame, min_year_gap: int, expected_author_ids: List[int]
) -> None:
    links = retain_affiliation_links_with_min_year_gap(link_gdf=link_df, min_year_gap=min_year_gap)
    assert links[ARTICLE_AUTHOR_ID_COLUMN].tolist() == expected_author_ids


def test_create_affiliation_links(
    article_df: pd.DataFrame, affiliation_gdf: gpd.GeoDataFrame
) -> None:
    article_author_df = get_articles_per_author(article_df=article_df)
    affiliation_map = get_affiliation_id_map(affiliation_gdf=affiliation_gdf)
    article_author_df = apply_parent_affiliation_id_and_idx(
        article_author_df=article_author_df, affiliation_map=affiliation_map
    )

    links = create_affiliation_links(
        article_author_df=article_author_df, affiliation_gdf=affiliation_gdf
    )
    assert links.from_node.tolist() == [8, 8, 10, 10]
    assert links.to_node.tolist() == [9, 9, 11, 11]


@pytest.mark.parametrize(
    "min_weight,expected_edge_df_length,expected_article_count_from, expected_article_count_to",
    [
        (0, 3, [2, 2, 2], [2, 1, 1]),
        (1, 3, [2, 2, 2], [2, 1, 1]),
        (2, 1, [2], [2]),
    ],
)
def test_create_graph_from_links(
    link_df: pd.DataFrame,
    min_weight: int,
    expected_edge_df_length: int,
    expected_article_count_from: List[int],
    expected_article_count_to: List[int],
) -> None:
    edge_graph = create_graph_from_links(link_gdf=link_df, min_weight=min_weight)
    edge_df = edge_graph.gdf
    assert len(edge_df) == expected_edge_df_length
    assert edge_df[ARTICLE_COUNT_COLUMN].tolist() == expected_article_count_from
    assert edge_df[f"{ARTICLE_COUNT_COLUMN}_to"].tolist() == expected_article_count_to
    assert edge_df[AFFILIATION_EDGE_COUNT_COLUMN].min() >= min_weight


@pytest.mark.parametrize(
    "min_weight,expected_from_nodes,expected_to_nodes,expected_edge_df_length,"
    "expected_article_count_from, expected_article_count_to",
    [
        (0, [8, 8, 10, 10], [9, 9, 11, 11], 2, [2, 2], [2, 2]),
        (1, [8, 8, 10, 10], [9, 9, 11, 11], 2, [2, 2], [2, 2]),
        (2, [8, 8, 10, 10], [9, 9, 11, 11], 2, [2, 2], [2, 2]),
        (3, [8, 8, 10, 10], [9, 9, 11, 11], 0, [], []),
    ],
)
def test_affiliation_network_processor(
    article_df: pd.DataFrame,
    affiliation_gdf: gpd.GeoDataFrame,
    min_weight: int,
    expected_from_nodes: List[int],
    expected_to_nodes: List[int],
    expected_edge_df_length: int,
    expected_article_count_from: List[int],
    expected_article_count_to: List[int],
) -> None:
    # Verifies the full network processing pipeline
    processor = AffiliationNetworkProcessor(article_df=article_df, affiliation_gdf=affiliation_gdf)

    links = processor.get_affiliation_links()
    assert links[FROM_NODE_COLUMN].tolist() == expected_from_nodes

    edge_graph = processor.get_affiliation_graph(min_edge_weight=min_weight)
    assert edge_graph is not None
    if edge_graph is not None:
        edge_df = edge_graph.gdf
        assert len(edge_df) == expected_edge_df_length
        assert edge_df[ARTICLE_COUNT_COLUMN].tolist() == expected_article_count_from
        assert edge_df[f"{ARTICLE_COUNT_COLUMN}_to"].tolist() == expected_article_count_to

        if expected_edge_df_length > 0:
            assert edge_df[AFFILIATION_EDGE_COUNT_COLUMN].min() >= min_weight
        else:
            assert math.isnan(edge_df[AFFILIATION_EDGE_COUNT_COLUMN].min())
