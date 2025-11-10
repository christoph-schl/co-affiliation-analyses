import os
from collections import namedtuple

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
ORG_TYPE_FILTER_LIST = ["univ", "resi", "hosp", "comp", "coll", "npo", "gov"]
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
    "univ": Color(128, 0, 128),
    "resi": Color(0, 106, 113),
    "hosp": Color(51, 78, 255),
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
    "hosp": "hosp",
    "comp": "comp",
    "resi": "resi",
    "npo": "npo",
    "univ|resi": "univ",
    "gov|resi": "resi",
    "univ": "univ",
    "comp|univ": "univ",
    "gov": "gov",
    "comp|resi": "resi",
    "meds|resi": "resi",
    "coll": "coll",
    "univ|hosp": "univ",
    "hosp|resi": "hosp",
    "hosp|univ": "univ",
    "gov|gov": "univ",
    "museum": "museum",
    "coll|resi": "coll",
    "npo|resi": "resi",
    "meds|npo": "npo",
    "milo": "milo",
    "other": "other",
}

SAMPLES_COLUMN = "samples"
