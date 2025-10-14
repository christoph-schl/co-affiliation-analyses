import os

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
