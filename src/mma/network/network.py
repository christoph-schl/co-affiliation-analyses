from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import structlog

from src.mma.constants import (
    ARTICLE_AFFILIATION_ID_COLUMN,
    ARTICLE_AFFILIATION_INDEX_COLUMN,
    UNMAPPED_AFFILIATION_ID_COLUMN,
)
from src.mma.network.utils import create_affiliation_links
from src.mma.utils.utils import (
    filter_links_by_country,
    get_affiliation_id_map,
    get_articles_per_author,
    validate_country_code,
)
from src.mma.utils.wrappers import get_execution_time

_logger = structlog.getLogger()

_ARTICLE_ID_COLUMN = "article_id"
_MULTIPLE_AFFILIATION_SEPARATOR = "-"


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
    affiliation_map: Dict[np.int64, np.int64] = None,
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
    affiliation_map: Dict[np.int64, np.int64] = None,
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


def _apply_parent_affiliation_id_and_idx(
    article_author_df: pd.DataFrame, affiliation_map: Dict[np.int64, np.int64]
):
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


@dataclass
class AffiliationNetworkProcessor:
    article_df: pd.DataFrame
    affiliation_gdf: gpd.GeoDataFrame
    _affiliation_map: Optional[Dict[np.int64, np.int64]] = field(init=False, default=None)

    """
   Processes article-author-affiliation data to generate affiliation link GeoDataFrames.

    This class handles authors with multiple affiliations and creates all possible
    unique links between their affiliations. Affiliation attributes, including geometries,
    are merged to produce a GeoDataFrame where each row represents a link between two
    affiliations with an associated line geometry. Optionally, the links can be filtered
    by ISO3 country codes. The resulting data can also be used to construct a networkx-compatible
    network for further network analysis.

    """

    def __post_init__(self) -> None:
        # compute the map after dataclass has been initialized
        self._affiliation_map = get_affiliation_id_map(affiliation_gdf=self.affiliation_gdf)

    @get_execution_time
    def get_affiliation_links(self, country_filter: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Creates a GeoDataFrame with all unique link combinations for authors with multiple
        affiliations.

        For each author in `self.article_df`, all possible pairwise links between affiliations
        stored in the `author_afids` column are generated. For example, if an author has
        affiliations `[1, 2, 3]`, the resulting links will be `(1, 2)`, `(1, 3)`, and `(2, 3)`.
        Affiliation information, including geometry, is merged into the link DataFrame, resulting in
        a GeoDataFrame with line geometries representing each link.

        :param country_filter: Optional ISO3 country code to filter links by country.
        :return: A GeoDataFrame containing links between affiliations, with a line geometry for each
                 link.
        """

        validate_country_code(code=country_filter)

        article_author_df = get_articles_per_author(article_df=self.article_df)

        _apply_parent_affiliation_id_and_idx(
            article_author_df=article_author_df, affiliation_map=self._affiliation_map
        )

        link_gdf = create_affiliation_links(
            affiliation_gdf=self.affiliation_gdf, article_author_df=article_author_df
        )

        if country_filter is not None:
            link_gdf = filter_links_by_country(link_gdf=link_gdf, country_filter=country_filter)

        return link_gdf
