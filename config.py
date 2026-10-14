"""
Central configuration for the BAAC road-safety pipeline.

Every script reads its paths and parameters from here. Paths can be moved
without editing code by setting the BAAC_ROOT environment variable
(e.g. ``export BAAC_ROOT=/data/baac``); by default everything lives inside the
repository folder.

Layout produced by the pipeline::

    <ROOT>/data/raw/<year>/{caracteristiques,lieux,vehicules,usagers}-<year>.csv
    <ROOT>/data/processed/parquet/   cleaned dataset (hive-partitioned by year)
    <ROOT>/data/processed/csv/       same data, one CSV per year
    <ROOT>/outputs/<step>/           figures and tables of each pipeline step
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("BAAC_ROOT", Path(__file__).resolve().parent))

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
RAW_DIR = Path(os.environ.get("BAAC_RAW_DIR", ROOT / "data" / "raw"))
PROCESSED_DIR = ROOT / "data" / "processed"
PARQUET_DIR = PROCESSED_DIR / "parquet"
CSV_DIR = PROCESSED_DIR / "csv"

OUTPUT_ROOT = ROOT / "outputs"
OVERVIEW_DIR = OUTPUT_ROOT / "00_data_overview"
RISK_FACTORS_DIR = OUTPUT_ROOT / "02_risk_factors"
SPATIAL_DIR = OUTPUT_ROOT / "03_spatial_segmentation"
PREDICTION_DIR = OUTPUT_ROOT / "04_severity_prediction"   # + "_<feature_set>" suffix

# File produced by step 03 and consumed by steps 04/05 (variant "geo").
SPATIAL_CLUSTERS_PATH = SPATIAL_DIR / "dbscan_clusters.parquet"
SPATIAL_CLUSTER_COL = "cluster_dbscan"

# ------------------------------------------------------------------
# General
# ------------------------------------------------------------------
YEARS = range(2005, 2025)
RANDOM_STATE = 0

# Official BAAC coding of `grav` is NOT ordered by severity
# (1=unharmed, 2=killed, 3=hospitalised, 4=light injury). It is recoded to an
# ordinal scale: 1=unharmed < 2=light injury < 3=hospitalised < 4=killed.
GRAV_TO_ORDINAL = {1: 1, 4: 2, 3: 3, 2: 4}
SEVERITY_LABELS = {1: "Unharmed", 2: "Light injury", 3: "Hospitalised", 4: "Killed"}
TARGET_COL = "gravite_ordinale"

# Label used for missing nominal categories in the cleaned dataset.
MISSING_LABEL = "Non renseigné"

# ------------------------------------------------------------------
# Step 01 - preprocessing
# ------------------------------------------------------------------
FILE_KEYWORDS = {
    # "carcteristiques" (sic): typo present in the official 2021/2022 files.
    "caracteristiques": ["caracteristiques", "caract", "carcteristiques"],
    "lieux": ["lieux"],
    "vehicules": ["vehicules", "vehicule"],
    "usagers": ["usagers", "usager"],
}
SENTINELS_GENERIC = ["-1", -1, ".", "", " "]

# Vehicle <-> user join key. From 2019 the files carry `id_vehicule`; before 2019
# the vehicle key is `Num_Veh` (column `num_veh`). Joining on `num_acc` alone for
# those years pairs every user with EVERY vehicle of the accident (row fan-out,
# wrong vehicle attributes). True (default, correct) = also join on `num_veh` when
# `id_vehicule` is absent. False = legacy behaviour that duplicated rows before 2019
# (kept only to reproduce the old results; see the "Known issue" section of the README).
JOIN_ON_NUM_VEH = True
# Continuous / ordinal variables imputed with IterativeImputer. lat/long are
# deliberately excluded (see scripts/01_preprocessing.py).
CONTINUOUS_OR_ORDINAL_COLS = ["an_nais", "vma", "nbv", "lartpc", "larrout", "age"]

# ------------------------------------------------------------------
# Step 02 - exploratory analysis and risk factors
# ------------------------------------------------------------------
NUMERIC_COLS = ["age", "heure_accident", "vma", "nbv", "lartpc", "larrout",
                "lat", "long", "jour_semaine"]
# 21 categorical circumstance factors (association rules and prediction).
CATEGORICAL_FACTORS = [
    "agg", "atm", "catr", "catv", "choc", "circ", "col", "infra", "int",
    "lum", "manv", "obs", "obsm", "plan", "prof", "sexe", "situ", "surf",
    "trajet", "vosp", "weekend",
]
ASSOC_SAMPLE_SIZE = 300_000
ASSOC_MIN_LIFT = 1.2
FPGROWTH_MIN_SUPPORT = 0.02
APRIORI_MIN_SUPPORT = 0.05
ASSOC_MAX_LEN = 4

# ------------------------------------------------------------------
# Step 03 - spatial segmentation
# ------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0088
EXPOSURE_FILE = None        # optional CSV with columns "com" (INSEE code), "population"
DBSCAN_SAMPLE_SIZE = 150_000
HDBSCAN_SAMPLE_SIZE = 100_000
ST_DBSCAN_SAMPLE_SIZE = 50_000
DBSCAN_MIN_SAMPLE_SIZE_RETRY = 20_000
DBSCAN_EPS_KM = 0.5
DBSCAN_MIN_SAMPLES = 15
HDBSCAN_MIN_CLUSTER_SIZE = 25
ST_EPS_SPATIAL_KM = 0.5
ST_EPS_TEMPORAL_HOURS = 2.0
ST_MIN_SAMPLES = 8

# Bounding boxes (lat_min, lat_max, long_min, long_max) of the territories
# covered by BAAC. One box per territory (instead of a single national one)
# rejects corrupted coordinates explicitly.
TERRITORIES = {
    "Mainland France (+ Corsica)": (41.0, 51.5, -5.5, 9.8),
    "Guadeloupe": (15.8, 16.6, -61.9, -60.9),
    "Martinique": (14.2, 15.0, -61.3, -60.7),
    "French Guiana": (2.0, 6.0, -54.7, -51.5),
    "Reunion": (-21.5, -20.7, 55.1, 55.9),
    "Mayotte": (-13.1, -12.5, 44.9, 45.4),
    "Saint-Pierre-et-Miquelon": (46.6, 47.3, -56.6, -56.0),
    "Saint-Martin / Saint-Barthelemy": (17.8, 18.2, -63.2, -62.7),
    "French Polynesia": (-28.0, -7.0, -155.0, -134.0),
    "New Caledonia": (-23.0, -19.0, 163.0, 168.5),
    "Wallis and Futuna": (-14.5, -13.0, -178.5, -176.0),
}
MAINLAND = "Mainland France (+ Corsica)"

# ------------------------------------------------------------------
# Steps 04/05 - severity prediction
# ------------------------------------------------------------------
TRAIN_YEARS = list(range(2005, 2022))   # 2005-2021
TEST_YEARS = list(range(2022, 2025))    # 2022-2024
CONTINUOUS_COLS = ["age", "heure_accident", "vma", "nbv", "lartpc",
                   "larrout", "jour_semaine"]

# Feature sets (see EXPERIMENTS.md, section "Feature-set selection").
EXTENDED_CATEGORICAL = [
    "equip_ceinture", "equip_casque", "equip_enfant", "equip_gilet",
    "equip_airbag_2rm", "equip_gants_2rm",
    "catu", "place", "motor", "senc", "mois", "locp", "actp", "etatp",
]
EXTENDED_CEINTURE_ONLY = [
    "equip_ceinture", "catu", "place", "motor", "senc", "mois",
    "locp", "actp", "etatp",
]
FEATURE_SETS = {
    "base": [],
    "extended": EXTENDED_CATEGORICAL,
    "extended_ceinture": EXTENDED_CEINTURE_ONLY,   # retained feature set
}
DEFAULT_FEATURE_SET = "extended_ceinture"

# Resource profiles. "conservative" is what produced every reported result
# (single-core, strongly subsampled, memory-safe on a Windows PC). "powerful"
# is provided for workstations with much more RAM/cores.
PROFILES = {
    "conservative": dict(
        n_jobs=1, rf_estimators=150, rf_depth=14, rf_max_samples=0.10,
        gb_estimators=250, gb_depth=7, gb_subsample=0.2, smote_max_rows=300_000),
    "powerful": dict(
        n_jobs=8, rf_estimators=300, rf_depth=20, rf_max_samples=0.5,
        gb_estimators=500, gb_depth=9, gb_subsample=0.7, smote_max_rows=1_000_000),
}
DEFAULT_PROFILE = "conservative"
SHAP_SAMPLE_SIZE = 5_000

# Robustness protocol (step 05)
ROBUSTNESS_SEEDS = [0, 1, 2, 3, 4]
ROBUSTNESS_MODELS = ["XGBoost-Ordinal", "XGBoost-SMOTE", "XGBoost"]
BOOTSTRAP_REPLICATES = 1000
