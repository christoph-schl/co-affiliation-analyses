from dataclasses import InitVar, dataclass, field
from functools import partial
from typing import Dict, List, Optional, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import structlog

from src.mma.constants import (
    ARTICLE_AFFILIATION_ID_COLUMN,
    ARTICLE_AFFILIATION_INDEX_COLUMN,
    LEVEL_2_CLASSIFICATION,
    ORGANISATION_TYPE_COLUMN,
    PREFERRED_AFFILIATION_NAME_COLUMN,
    UNMAPPED_AFFILIATION_ID_COLUMN,
)
from src.mma.dataframe.models.affiliation import AffiliationSchema
from src.mma.dataframe.models.article import ArticleSchema
from src.mma.network.utils import (
    Edge,
    create_affiliation_links,
    create_graph_from_links,
    get_vos_cluster_numbers,
    retain_affiliation_links_with_min_year_gap,
)
from src.mma.utils.utils import (
    get_affiliation_id_map,
    get_articles_per_author,
    get_link_nodes,
    validate_country_code,
)
from src.mma.utils.wrappers import get_execution_time, parallelize_dataframe

_logger = structlog.getLogger()

_ARTICLE_ID_COLUMN = "article_id"
_MULTIPLE_AFFILIATION_SEPARATOR = "-"
_ORG_TYPE_OTHER = "other"
_NODE_ID_COLUMN = "node"  # corresponds to `affiliation_id`


class MissingEdgeGraphError(RuntimeError):
    """Raised when an operation requires an edge graph but none is present."""


def _get_parent_id(
    affiliation_id: str, affiliation_map: Dict[np.int64, np.int64]
) -> List[np.int64]:

    affiliation_ids = list(
        np.asarray(affiliation_id.split(_MULTIPLE_AFFILIATION_SEPARATOR)).astype(np.int64)
    )
    affiliation_ids = np.asarray(
        list(set([affiliation_map.get(aid, aid) for aid in affiliation_ids]))
    )
    return affiliation_ids


def _get_parent_affiliation_idx(
    affiliation_id: str,
    affiliation_map: Dict[np.int64, np.int64],
) -> Dict[np.int64, np.int64]:
    """
    Determines the parent affiliation indexes.
    :param affiliation_id: String with multitple affiliation IDs separated by '-'.
    :param affiliation_map: Mapping of affiliation IDs to parent IDs.
    :return: A dictionary with the parent affiliation IDs as keys and the indexes as values.
    """
    affiliation_ids = _get_parent_id(affiliation_id=affiliation_id, affiliation_map=affiliation_map)
    sorted_ids = sorted(affiliation_ids)
    sorted_idx_dict = dict(zip(sorted_ids, list(np.argsort(affiliation_ids))))
    return sorted_idx_dict


def _get_parent_affiliation_id(
    affiliation_id: str,
    affiliation_map: Dict[np.int64, np.int64],
) -> Union[List[np.int64], Dict[np.int64, np.int64]]:
    """
    Determines the parent affiliation IDs.
    :param affiliation_id: String with multitple affiliation IDs separated by '-'.
    :param affiliation_map: Mapping of affiliation IDs to parent IDs.
    :return: A List with parent affiliation IDs if found.
    """

    affiliation_ids = _get_parent_id(affiliation_id=affiliation_id, affiliation_map=affiliation_map)
    sorted_ids = sorted(affiliation_ids)

    return list(np.array(sorted_ids).astype(np.int64))


def apply_parent_affiliation_id_and_idx(
    article_author_df: pd.DataFrame,
    affiliation_map: Union[Dict[np.int64, np.int64], Dict[int, int]],
) -> pd.DataFrame:
    """
    Determine the parent affiliation IDs for each author and update the corresponding columns.

    The function adds affiliation id and index information to the input DataFrame:
    - `author_afids`: contains the resolved (parent) affiliation IDs.
    - `affiliation_idx`: contains a dictionary mapping each parent affiliation ID to its index.
    The original affiliation IDs are preserved in a new column named `unmapped_id`.

    :param article_author_df:
        DataFrame containing author and affiliation information.
    :param affiliation_map:
        Dictionary mapping affiliation IDs to their corresponding parent IDs.
    :return:
        None. The function modifies `article_author_df` in place.
    """

    article_author_df = article_author_df.copy()

    aff_col = ARTICLE_AFFILIATION_ID_COLUMN
    idx_col = ARTICLE_AFFILIATION_INDEX_COLUMN
    unmapped_col = UNMAPPED_AFFILIATION_ID_COLUMN

    # keep the original affiliation IDs
    article_author_df[unmapped_col] = article_author_df[aff_col]

    # safe partials for the two variants of the helper
    get_parent_idx = partial(_get_parent_affiliation_idx, affiliation_map=affiliation_map)
    get_parent = partial(_get_parent_affiliation_id, affiliation_map=affiliation_map)

    # get parent affiliation id and parent affiliation id idx
    article_author_df[idx_col] = article_author_df[aff_col].apply(get_parent_idx)
    article_author_df[aff_col] = article_author_df[aff_col].apply(get_parent)
    return article_author_df


def _check_int_value(value: int) -> None:
    if not isinstance(value, int):
        raise ValueError("value must be an integer")
    if value < 0:
        raise ValueError("value must be non-negative")


@dataclass
class AffiliationNetworkProcessor:
    """
    Processes article-author-affiliation data to generate affiliation link GeoDataFrames.

    This class handles authors with multiple affiliations and creates all possible
    unique links between their affiliations. Affiliation attributes, including geometries,
    are merged to produce a GeoDataFrame where each row represents a link between two
    affiliations with an associated line geometry. Optionally, the links can be filtered
    by ISO3 country codes. The resulting data can also be used to construct a networkx-compatible
    network for further network analysis.

    Attributes
    ----------
    article_df : pandas.DataFrame, optional
        Source articles DataFrame validated with `ArticleSchema`.
        Required if `link_gdf` is not supplied.

    affiliation_gdf : geopandas.GeoDataFrame
        Source affiliations GeoDataFrame validated with `AffiliationSchema`.

    country_filter : str, optional
        ISO3 country filter applied when generating links.
        Defaults to ``None`` (no filter).

    link_gdf : geopandas.GeoDataFrame, optional
        GeoDataFrame containing links between affiliations, each represented by a line geometry.
        Can be provided to avoid expensive computations.

    """

    # source inputs (optional if link_gdf is provided)
    article_df: Optional[pd.DataFrame] = None
    affiliation_gdf: Optional[gpd.GeoDataFrame] = None

    # optional country filter
    country_filter: Optional[str] = None

    # accept an initial link_gdf at init time but don't bind it directly as a field
    link_gdf: InitVar[Optional[gpd.GeoDataFrame]] = None

    _link_gdf: Optional[gpd.GeoDataFrame] = None
    _edge_graph: Optional[Edge] = None
    _min_edge_weight: Optional[int] = 0
    _min_year_gap: Optional[int] = 0
    _article_author_df: pd.DataFrame = field(init=False, default=None)
    _vos_org_type_colors: pd.DataFrame = field(init=False, default=None)
    _affiliation_map: Optional[Dict[np.int64, np.int64]] = field(init=False, default=None)

    def __post_init__(self, link_gdf: Optional[gpd.GeoDataFrame]) -> None:

        validate_country_code(code=self.country_filter)

        if link_gdf is not None:
            self._link_gdf = link_gdf
            # do not prepare article/affiliation derived structures — we can skip expensive work
            return

        if self.article_df is None or self.affiliation_gdf is None:
            raise ValueError(
                "Either provide both `article_df` and `affiliation_gdf` or `link_gdf`."
            )

        self._validate_input_dataframes()

        self._affiliation_map = get_affiliation_id_map(affiliation_gdf=self.affiliation_gdf)

        self._article_author_df = get_articles_per_author(article_df=self.article_df)

        self._article_author_df = parallelize_dataframe(
            input_function=apply_parent_affiliation_id_and_idx,
            df=self._article_author_df,
            affiliation_map=self._affiliation_map,
        )

    @property
    def min_edge_weight(self) -> Optional[int]:
        return self._min_edge_weight

    @min_edge_weight.setter
    def min_edge_weight(self, value: int) -> None:
        _check_int_value(value=value)
        self._min_edge_weight = value

    @property
    def min_year_gap(self) -> Optional[int]:
        return self._min_year_gap

    @min_year_gap.setter
    def min_year_gap(self, value: int) -> None:
        _check_int_value(value=value)
        self._min_year_gap = value

    @property
    def vos_org_type_colors(self) -> pd.DataFrame:
        if self._vos_org_type_colors is None:
            if self._edge_graph is None:
                raise MissingEdgeGraphError(
                    "Edge graph is not set. "
                    "Set it when constructing or call `_create_affiliation_graph()` first."
                )
            self._vos_org_type_colors = get_vos_cluster_numbers(
                affiliation_graph=self._edge_graph.graph
            )
        return self._vos_org_type_colors

    @property
    def edge(self) -> Edge:
        if self._edge_graph is None:
            raise MissingEdgeGraphError(
                "Edge graph is not set. "
                "Set it when constructing or call `_create_affiliation_graph()` first."
            )
        return self._edge_graph

    @edge.setter
    def edge(self, value: Edge) -> None:
        self._edge_graph = value

    @property
    def link(self) -> gpd.GeoDataFrame:
        return self._link_gdf

    @link.setter
    def link(self, value: gpd.GeoDataFrame) -> None:
        self._link_gdf = value

    def _validate_input_dataframes(self) -> None:
        self.article_df = ArticleSchema.validate(self.article_df)
        self.affiliation_gdf = AffiliationSchema.validate(self.affiliation_gdf)

    @get_execution_time
    def get_affiliation_links(self, min_year_gap: Optional[int] = None) -> gpd.GeoDataFrame:
        """
        Creates a GeoDataFrame with all unique link combinations for authors with multiple
        affiliations.

        For each author in `self.article_df`, all possible pairwise links between affiliations
        stored in the `author_afids` column are generated. For example, if an author has
        affiliations `[1, 2, 3]`, the resulting links will be `(1, 2)`, `(1, 3)`, and `(2, 3)`.
        Affiliation information, including geometry, is merged into the link DataFrame, resulting in
        a GeoDataFrame with line geometries representing each link.

        :param min_year_gap:
                Minimum required difference (in years) between earliest and latest publication
                years for an author at the same affiliation pair. If < 1, the original DataFrame
                is returned.

        :return:
                A GeoDataFrame containing links between affiliations, with a line geometry for each
                link.
        """

        self._create_affiliation_links(min_year_gap=min_year_gap)

        return self._link_gdf

    def get_affiliation_graph(self, min_edge_weight: Optional[int] = None) -> Optional[Edge]:
        """
        Creates an affiliation network graph from a GeoDataFrame of links between nodes
        (affiliations).

        This function aggregates link data between pairs of nodes (`from_node`, `to_node`), computes
        the connection strength for each (source, target) pair, and returns a weighted NetworkX
        graph. Optionally, weak connections (below a given weight threshold) can be filtered out.

        :param min_edge_weight: The minimum link strength to retain. Edges with lower weights are
                                removed from the graph. If ``None``, all edges are included.

        :return: A weighted, undirected NetworkX graph where:

           - **Nodes** represent affiliations.
           - **Edges** connect affiliations based on authors with multiple affiliations.
           - **Edge weights** correspond to the strength or frequency of those links.
        """

        self._create_affiliation_graph(min_edge_weight=min_edge_weight)
        return self._edge_graph

    def _create_affiliation_links(self, min_year_gap: Optional[int] = None) -> None:
        if self._link_gdf is None:
            self._link_gdf = create_affiliation_links(
                affiliation_gdf=self.affiliation_gdf,
                article_author_df=self._article_author_df,
                country_filter=self.country_filter,
            )
            self.link = self._link_gdf

        if min_year_gap is not None:
            self.min_year_gap = min_year_gap

        self._link_gdf = retain_affiliation_links_with_min_year_gap(
            link_gdf=self._link_gdf, min_year_gap=self.min_year_gap
        )
        self._reclassify_link_org_types()
        self.link = self._link_gdf

    def _create_affiliation_graph(self, min_edge_weight: Optional[int] = None) -> None:

        if min_edge_weight is not None:
            self.min_edge_weight = min_edge_weight

        # generate links between affiliations if not cached
        self._create_affiliation_links()

        if self._edge_graph is None:
            self._edge_graph = create_graph_from_links(
                link_gdf=self._link_gdf, min_weight=self.min_edge_weight
            )

            self._add_org_type_to_graph_and_relabel_nodes()

            self.edge = self._edge_graph

    def _reclassify_link_org_types(self) -> None:
        self.link.loc[self.link[ORGANISATION_TYPE_COLUMN].isnull(), ORGANISATION_TYPE_COLUMN] = (
            _ORG_TYPE_OTHER
        )
        self.link[ORGANISATION_TYPE_COLUMN] = self.link[ORGANISATION_TYPE_COLUMN].apply(
            lambda x: LEVEL_2_CLASSIFICATION[x]
        )

        org_type_to = f"{ORGANISATION_TYPE_COLUMN}_to"
        if org_type_to in self.link.columns:
            self.link[org_type_to] = self.link[org_type_to].apply(
                lambda x: LEVEL_2_CLASSIFICATION[x]
            )

    def _add_org_type_to_graph_and_relabel_nodes(self) -> None:

        nodes = get_link_nodes(link_gdf=self._link_gdf)[
            [_NODE_ID_COLUMN, ORGANISATION_TYPE_COLUMN, PREFERRED_AFFILIATION_NAME_COLUMN]
        ]

        self._add_org_type_attributes_to_graph(node_df=nodes)

        self._relabel_graph_nodes(node_df=nodes)

    def _relabel_graph_nodes(self, node_df: gpd.GeoDataFrame) -> None:
        if self._edge_graph is not None:
            name_map = (
                node_df[[_NODE_ID_COLUMN, PREFERRED_AFFILIATION_NAME_COLUMN]]
                .set_index(_NODE_ID_COLUMN)[PREFERRED_AFFILIATION_NAME_COLUMN]
                .to_dict()
            )

            self._edge_graph.graph = nx.relabel_nodes(self._edge_graph.graph, name_map)

    def _add_org_type_attributes_to_graph(self, node_df: gpd.GeoDataFrame) -> None:
        if self._edge_graph is not None:
            org_map = (
                node_df[[_NODE_ID_COLUMN, ORGANISATION_TYPE_COLUMN]]
                .set_index(_NODE_ID_COLUMN)[ORGANISATION_TYPE_COLUMN]
                .to_dict()
            )
            for node, data in self._edge_graph.graph.nodes(data=True):
                data[ORGANISATION_TYPE_COLUMN] = org_map[node]
