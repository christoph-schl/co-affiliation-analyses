import geopandas as gpd
import pandas as pd
import pytest
from shapely import Point


def get_article_test_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doi": ["Doi1", "Doi2", "Doi3", "Doi4"],
            "eid": ["Eid1", "Eid2", "Eid3", "Eid4"],
            "cover_date": ["2013-05-01", "2015-05-01", "2020-05-01", "2023-05-01"],
            "author_ids": ["Author1", "Author1", "Author2", "Author1;Author3"],
            "author_afids": ["1-2-3", "1-2-3", "4-5", "4-5;8"],
        }
    )


def get_affiliation_test_df() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "affiliation_id": [1, 2, 3, 4, 5, 8, 9, 10, 11],
            "affiliation_id_parent": [8, 9, 8, 10, 11, 8, 9, 10, 11],
            "preferred_name": [
                "Aff1",
                "Aff2",
                "Aff3",
                "Aff4",
                "Aff5",
                "Aff6",
                "Aff7",
                "Aff8",
                "Aff9",
            ],
            "org_type": ["univ", "univ", "resi", "resi", "gov", "univ", "resi", "resi", "gov"],
            "geometry": [
                Point(16.3738, 48.2082),  # Wien
                Point(16.3738, 48.2082),  # Wien
                Point(15.4395, 47.0707),  # Graz
                Point(15.4395, 47.0707),  # Graz
                Point(11.4041, 47.2692),  # Innsbruck
                Point(16.3738, 48.2082),  # Wien
                Point(15.4395, 47.0707),  # Graz
                Point(15.4395, 47.0707),  # Graz
                Point(11.4041, 47.2692),  # Innsbruck
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def get_impact_test_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["Eid1", "Eid2", "Eid3"],
            "hazen_perc_med": [75.0248, 97.7975, 99.7569],
            "class_name": [
                "{Surgery,Oral Surgery,Otorhinolaryngology}",
                "{Dermatology}",
                "{Industrial Relations,Engineering (all),"
                "Computer Science Applications,Strategy and Management,Law}",
            ],
        }
    )


def get_link_df_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eid": ["Eid1", "Eid2", "Eid3", "Eid4"],
            "from_node": [8, 8, 10, 10],
            "to_node": [9, 9, 11, 1],
            "preferred_name": ["Univ_8", "Univ_8", "Resi_10", "Resi_10"],
            "preferred_name_to": ["Univ_9", "Univ_9", "Resi_11", "Resi_1"],
            "cover_date": ["2013-05-01", "2015-05-01", "2020-05-01", "2023-05-01"],
            "author_ids": ["Author1", "Author1", "Author2", "Author3"],
        }
    )


@pytest.fixture(name="article_df")
def fixture_article_df() -> pd.DataFrame:
    return get_article_test_df()


@pytest.fixture(name="affiliation_gdf")
def fixture_affiliation_gdf() -> gpd.GeoDataFrame:
    return get_affiliation_test_df()


@pytest.fixture(name="impact_df")
def fixture_impact_df() -> pd.DataFrame:
    return get_impact_test_df()


@pytest.fixture(name="link_df")
def fixture_link_df() -> pd.DataFrame:
    return get_link_df_test()
