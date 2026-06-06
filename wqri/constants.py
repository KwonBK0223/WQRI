"""Shared feature definitions for WQRI experiments."""

# Public CSV schema after English column-name conversion.
RAW_LOCATION_COLUMN = "Observation_point"
RAW_DATE_COLUMN = "Observation_date"

# Internal columns used after preprocessing.
INTERNAL_LOCATION_COLUMN = "Location"
NODE_ID_COLUMN = "node_id"
SEQUENCE_COLUMN = "seq"

RAW_FEATURE_COLUMNS = [
    "Water_temperature", "DO", "BOD", "COD", "SS", "TN", "TP", "TOC", "pH", "EC",
]

ENV_FEATURE_COLUMNS = [
    "Water_temperature_scaled", "DO_scaled", "BOD_scaled", "COD_scaled", "SS_scaled",
    "TN_scaled", "TP_scaled", "TOC_scaled", "pH_scaled", "EC_scaled",
]

RISK_FEATURE_COLUMNS = [
    "Water_temperature_type1", "DO_type2", "BOD_type1", "COD_type1", "SS_type1",
    "TN_type1", "TP_type1", "TOC_type1", "pH_type3", "EC_type1",
]

# The user's earlier preprocessing notebook used a few spelling variants.
# These aliases make the public code robust to those already-created CSV files.
COLUMN_ALIASES = {
    "Water_temparature_scaled": "Water_temperature_scaled",
    "Water_temparature_type1": "Water_temperature_type1",
    "Water_temprature_scaled": "Water_temperature_scaled",
    "Water_temprature_type1": "Water_temperature_type1",
    "Water_tempratrue_type3": "Water_temperature_type3",
}

REQUIRED_INPUT_COLUMNS = [
    RAW_LOCATION_COLUMN,
    RAW_DATE_COLUMN,
    *RAW_FEATURE_COLUMNS,
    *ENV_FEATURE_COLUMNS,
    *RISK_FEATURE_COLUMNS,
]

PREPARED_COLUMNS = [
    INTERNAL_LOCATION_COLUMN,
    SEQUENCE_COLUMN,
    NODE_ID_COLUMN,
    *RAW_FEATURE_COLUMNS,
    *ENV_FEATURE_COLUMNS,
    *RISK_FEATURE_COLUMNS,
]

DEFAULT_WINDOW_LIST = [3, 5, 7, 10, 14, 20, 30]
