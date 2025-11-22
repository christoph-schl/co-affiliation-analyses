import os
from collections import namedtuple
from pathlib import Path

# affiliation column names
AFFILIATION_ID_COLUMN = "affiliation_id"
AFFILIATION_ID_PARENT_COLUMN = "affiliation_id_parent"
GEOMETRY_COLUMN = "geometry"
WGS84_EPSG = 4326
ISO3_COUNTRY_CODE_COLUMN = "iso3_code"
PREFERRED_AFFILIATION_NAME_COLUMN = "preferred_name"
ORGANISATION_TYPE_COLUMN = "org_type"


# article column names
ARTICLE_AFFILIATION_ID_COLUMN = "author_afids"
UNMAPPED_AFFILIATION_ID_COLUMN = "unmapped_id"
ARTICLE_AFFILIATION_INDEX_COLUMN = "affiliation_idx"
ARTICLE_AFFILIATION_COUNT_COLUMN = "affiliation_count"
AFFILIATION_NAME_VARIANTS_COLUMN = "name_variants"
COVER_DATE_COLUMN = "cover_date"
ARTICLE_AUTHOR_ID_COLUMN = "author_ids"


# link or edge column names
FROM_NODE_COLUMN = "from_node"
TO_NODE_COLUMN = "to_node"
FROM_AFFILIATION_INDEX_COLUMN = "from_affiliation_idx"
TO_AFFILIATION_INDEX_COLUMN = "to_affiliation_idx"
AFFILIATION_EDGE_COUNT_COLUMN = "affiliation_edge_count"

# constants for parallel processing
DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING: int = max((os.cpu_count() or 2) - 2, 1)
CHUNK_SIZE_PARALLEL_PROCESSING = 400

# znib columns
TRAVEL_TIME_SEC_COLUMN = "duration_s"
ARTICLE_COUNT_COLUMN = "article_count"
LN_PROD_ARTICLE_COUNT_COLUMN = "ln_prod_article_count"
LN_DURATION_COLUMN = "ln_duration"
ORG_TYPE_FILTER_LIST = ["uni", "res", "med", "comp", "coll", "npo", "gov"]
DEFAULT_REGULARIZATION_STRENGTH = 3.0
DEFAULT_ZERO_INFLATION_MODEL = "logit"
DEFAULT_DISPERSION_POWER = 2
DEFAULT_FIT_METHOD = "bfgs"
DEFAULT_MAX_ITERATIONS = 400

# valhalla
DEFAULT_VALHALLA_BASE_URL = "http://localhost:8002"
DURATION_S_COLUMN = "duration_s"
DISTANCE_M_COLUMN = "distance_m"

# routing
AFFILIATION_ID_FROM_COLUMN = "affiliation_id_from"
AFFILIATION_ID_TO_COLUMN = "affiliation_id_to"

# impact
HAZEN_PERCENTILE_COLUMN = "hazen_perc_med"
CLASS_NAME_COLUMN = "class_name"
ITEM_ID_COLUMN = "item_id"  # corresponds to `eid`
MWPR_COLUMN = "mwPR"

# plot
AFFILIATION_CLASS_COLUMN = "affiliation_class"

Color = namedtuple("Color", ["red", "green", "blue"])
ORG_TYPE_COLORS = {
    "uni": Color(128, 0, 128),
    "res": Color(0, 106, 113),
    "med": Color(51, 78, 255),
    "comp": Color(70, 236, 250),
    "coll": Color(255, 127, 0),
    "npo": Color(34, 153, 84),
    "museum": Color(156, 34, 227),
    "gov": Color(242, 36, 17),
    "milo": Color(178, 34, 34),
    "other": Color(128, 128, 128),
}

EID_COLUMN = "eid"

# link organisation type reclassification
LEVEL_2_CLASSIFICATION = {
    "hosp": "med",
    "comp": "comp",
    "resi": "res",
    "npo": "npo",
    "univ|resi": "uni",
    "gov|resi": "res",
    "univ": "uni",
    "comp|univ": "uni",
    "gov": "gov",
    "comp|resi": "res",
    "meds|resi": "res",
    "coll": "coll",
    "univ|hosp": "uni",
    "hosp|resi": "med",
    "hosp|univ": "uni",
    "gov|gov": "uni",
    "museum": "museum",
    "coll|resi": "coll",
    "npo|resi": "res",
    "meds|npo": "npo",
    "milo": "milo",
    "other": "other",
    "uni": "uni",
    "res": "res",
    "med": "med",
}

SAMPLES_COLUMN = "samples"
NETWORK_COUNTRY = "AUT"

LINK_DIR = Path("network/links")
VOS_DIR = Path("network/vos")
GRAVITY_INPUT_DIR = Path("gravity/input")
GRAVITY_OUTPUT_DIR = Path("gravity/output")
PLOTS_OUTPUT_DIR = Path("plot/impact")

AFFILIATION_LINKS_PREFIX = "affiliation_links"
VOS_MAP_PREFIX = "map"
VOS_NETWORK_PREFIX = "network"
GRAVITY_INPUT_PREFIX = "model_input"
GRAVITY_INTRA_RESULTS_PREFIX = "intra_results"
GRAVITY_INTER_RESULTS_PREFIX = "inter_results"
VIOLINE_PLOT_FILENAME = "violine.png"
BAR_PLOT_FILENAME = "bar_plot.png"
TIMESERIES_PLOT_ORG_TYPE_FILENAME = "timeseries_org_type.png"
TIMESERIES_PLOT_INSTITUTION_FILENAME = "timeseries_institution.png"

CO_AFF_ALL_DATASET_NAME = "all"
CO_AFF_STABLE_DATASET_NAME = "stable"
