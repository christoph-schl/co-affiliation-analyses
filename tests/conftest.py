import geopandas as gpd
import pandas as pd
import pytest
from shapely import Point


def get_article_test_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doi": ["Doi1", "Doi2", "Doi3"],
            "eid": ["Eid1", "Eid2", "Eid3"],
            "cover_date": ["2013-05-01", "2015-05-01", "2020-05-01"],
            "author_ids": ["Author1", "Author1", "Author2"],
            "author_afids": ["1-2-3", "1-2-3", "4-5"],
        }
    )


def get_affiliation_test_df() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "affiliation_id": [1, 2, 3, 4, 5],
            "preferred_name": ["Aff1", "Aff2", "Aff3", "Aff4", "Aff5"],
            "org_type": ["univ", "univ", "resi", "resi", "gov"],
            "geometry": [
                Point(16.3738, 48.2082),  # Wien
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


@pytest.fixture(name="article_df")
def fixture_article_df() -> pd.DataFrame:
    return get_article_test_df()


@pytest.fixture(name="affiliation_gdf")
def fixture_affiliation_gdf() -> gpd.GeoDataFrame:
    return get_affiliation_test_df()


@pytest.fixture(name="impact_df")
def fixture_impact_df() -> pd.DataFrame:
    return get_impact_test_df()
