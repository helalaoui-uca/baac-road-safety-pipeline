"""
Severity-prediction toolkit shared by scripts/04_severity_prediction.py and
scripts/05_robustness.py: feature construction, temporal split, models
(including the ordinal and SMOTENC variants), evaluation metrics, SHAP.

Target: ordinal severity 1 unharmed < 2 light injury < 3 hospitalised < 4 killed.
The models are EXPLANATORY: some circumstance variables (collision type,
obstacle struck) are only known after the accident.
"""

import gc

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)

import config as cfg
from baac.common import encode_severity, load_master, log

N_CLASSES = 4
KILLED = 4
ALL_MODELS = ["RandomForest", "XGBoost", "XGBoost-Ordinal", "XGBoost-SMOTE", "LightGBM"]


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def required_columns(feature_set):
    extra = cfg.FEATURE_SETS[feature_set]
    return sorted(set(cfg.CATEGORICAL_FACTORS + cfg.CONTINUOUS_COLS + extra
                      + ["grav", "num_acc", "annee_dataset"]))


def load_prediction_master(feature_set):
    """Load only the columns the models need (instead of all ~70), then add
    the ordinal target."""
    return encode_severity(load_master(columns=required_columns(feature_set)))


def get_extra_cols(df, feature_set):
    wanted = cfg.FEATURE_SETS[feature_set]
    present = [c for c in wanted if c in df.columns]
    absent = [c for c in wanted if c not in df.columns]
    if wanted:
        log(f"[features] feature set '{feature_set}': added columns = {present}")
        if absent:
            log(f"[features] requested but absent from the dataset: {absent}")
    return present


def build_features(df, feature_cols):
    """Integer-code the categorical variables via pandas' 'category' dtype (far
    lighter than astype(str) + OrdinalEncoder on millions of rows) and
    downcast continuous variables to float32. Categorical columns are placed
    FIRST: SMOTENC needs to know which leading columns not to interpolate."""
    log(f"Building features on {len(df):,} rows...")
    cat_cols = [c for c in cfg.CATEGORICAL_FACTORS if c in feature_cols]
    num_cols = [c for c in cfg.CONTINUOUS_COLS if c in feature_cols]
    extra_cols = [c for c in feature_cols if c not in cat_cols and c not in num_cols]

    X = pd.DataFrame(index=df.index)
    for c in cat_cols + extra_cols:
        X[c] = df[c].astype("category").cat.codes.astype("int32")
    for c in num_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    if num_cols:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    return X, cat_cols, num_cols, extra_cols


def temporal_split(df):
    """Train on the oldest vintages, test on the most recent ones (no random
    split): BAAC has nomenclature breaks and non-stationary distributions."""
    if "annee_dataset" not in df.columns:
        raise KeyError("Column 'annee_dataset' is required for the temporal split.")
    year = pd.to_numeric(df["annee_dataset"], errors="coerce")
    train, test = year.isin(cfg.TRAIN_YEARS), year.isin(cfg.TEST_YEARS)
    print(f"Temporal split: {train.sum():,} training rows "
          f"({min(cfg.TRAIN_YEARS)}-{max(cfg.TRAIN_YEARS)}), {test.sum():,} test rows "
          f"({min(cfg.TEST_YEARS)}-{max(cfg.TEST_YEARS)}).")
    return train, test


def prepare_variant(master, variant, feature_set):
    """Return X_train, y_train, X_test, y_test, n_cat_cols for a data variant.

    "full": all rows, no spatial variable.
    "geo":  rows whose accident belongs to the DBSCAN sample (see step 03), with
            `cluster_dbscan` as an extra categorical variable (-1 = noise).
    """
    df = master.dropna(subset=[cfg.TARGET_COL])
    feature_cols = (cfg.CATEGORICAL_FACTORS + get_extra_cols(master, feature_set)
                    + cfg.CONTINUOUS_COLS)
    if variant == "geo":
        path = cfg.SPATIAL_CLUSTERS_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found: run scripts/03_spatial_segmentation.py first.")
        clusters = pd.read_parquet(path)
        col = cfg.SPATIAL_CLUSTER_COL
        missing = {"num_acc", "annee_dataset", col} - set(clusters.columns)
        if missing:
            raise KeyError(f"Missing columns in the cluster file: {missing}")
        log(f"Cluster file: {len(clusters):,} accidents.")
        df = df.merge(clusters[["num_acc", "annee_dataset", col]],
                      on=["num_acc", "annee_dataset"], how="inner")
        del clusters
        log(f"Geolocated subset with a spatial cluster: {len(df):,} user rows.")
        feature_cols = feature_cols + [col]
    elif variant != "full":
        raise ValueError("variant must be 'full' or 'geo'.")

    X, cat_cols, _, extra_cols = build_features(df, feature_cols)
    n_cat = len(cat_cols) + len(extra_cols)
    y = df[cfg.TARGET_COL].astype("int8")
    train, test = temporal_split(df)
    X_train, y_train = X[train.values], y[train.values]
    X_test, y_test = X[test.values], y[test.values]
    del df, X
    gc.collect()
    return X_train, y_train, X_test, y_test, n_cat


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
class OrdinalXGBClassifier:
    """Ordinal classification by binary decomposition (Frank & Hall, 2001).

    Severity is an ORDERED scale that standard softmax classifiers treat as
    purely nominal. K-1 binary XGBoost classifiers are trained, the k-th
    estimating P(severity > k). Class probabilities are rebuilt as successive
    differences of the cumulative probabilities.

    Each binary sub-problem is itself imbalanced, so it is re-weighted with
    scale_pos_weight, DAMPENED to the square root of the imbalance ratio: a full
    ratio applied independently at each threshold mis-calibrates the cumulative
    probabilities and, once recombined, over-predicts the severe classes
    (observed: macro-F1 below the plain multiclass XGBoost)."""

    def __init__(self, classes, **xgb_params):
        self.classes_sorted = sorted(classes)
        self.thresholds = self.classes_sorted[:-1]
        self.xgb_params = xgb_params
        self.models = {}

    def fit(self, X, y):
        from xgboost import XGBClassifier
        y = np.asarray(y)
        for k in self.thresholds:
            y_bin = (y > k).astype(int)
            pos_rate = max(y_bin.mean(), 1e-6)
            spw = np.sqrt((1 - pos_rate) / pos_rate)
            model = XGBClassifier(**self.xgb_params, objective="binary:logistic",
                                  eval_metric="logloss", scale_pos_weight=spw)
            model.fit(X, y_bin)
            self.models[k] = model
        return self

    def predict_proba(self, X):
        p_greater = np.column_stack(
            [self.models[k].predict_proba(X)[:, 1] for k in self.thresholds])
        # Independent estimates may locally violate monotonicity: enforce a
        # decreasing P(y > k) so class probabilities stay non-negative.
        p_greater = np.minimum.accumulate(p_greater, axis=1)
        n = len(self.classes_sorted)
        proba = np.zeros((X.shape[0], n))
        proba[:, 0] = 1 - p_greater[:, 0]
        for i in range(1, n - 1):
            proba[:, i] = p_greater[:, i - 1] - p_greater[:, i]
        proba[:, n - 1] = p_greater[:, n - 2]
        proba = np.clip(proba, 0, None)
        sums = proba.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return proba / sums

    def predict(self, X):
        return np.array(self.classes_sorted)[np.argmax(self.predict_proba(X), axis=1)]


class SMOTEXGBClassifier:
    """Multiclass XGBoost trained on a SMOTENC-rebalanced training set.

    SMOTENC (not plain SMOTE) because most variables are categorical codes:
    plain SMOTE would interpolate between arbitrary integer codes (category 3
    and 7 do not average to a meaningful "category 5"). SMOTENC treats
    categorical columns separately (mode of the nearest neighbours).
    Nearest-neighbour search becomes very costly on millions of rows, so a
    stratified subsample of at most `max_rows_before_smote` rows is drawn first.
    SMOTENC is only ever applied to training data."""

    def __init__(self, categorical_features, max_rows_before_smote=300_000,
                 random_state=0, **xgb_params):
        self.categorical_features = categorical_features
        self.max_rows_before_smote = max_rows_before_smote
        self.random_state = random_state
        self.xgb_params = xgb_params
        self.model = None
        self._label_offset = 0

    def fit(self, X, y):
        from imblearn.over_sampling import SMOTENC
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier

        y_arr = np.asarray(y)
        X_sub, y_sub = X, y_arr
        if len(X) > self.max_rows_before_smote:
            frac = self.max_rows_before_smote / len(X)
            X_sub, _, y_sub, _ = train_test_split(
                X, y_arr, train_size=frac, stratify=y_arr, random_state=self.random_state)
            log(f"[SMOTE] Stratified subsample of {len(X_sub):,} rows before SMOTENC "
                f"(limit: {self.max_rows_before_smote:,}).")
        log(f"[SMOTE] Applying SMOTENC on {len(X_sub):,} rows...")
        smote = SMOTENC(categorical_features=self.categorical_features,
                        random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X_sub, y_sub)
        classes, counts = np.unique(y_res, return_counts=True)
        log(f"[SMOTE] Rebalanced set: {len(X_res):,} rows "
            f"(classes: {dict(zip(classes.tolist(), counts.tolist()))}).")
        self._label_offset = int(y_res.min())
        self.model = XGBClassifier(**self.xgb_params, objective="multi:softprob",
                                   eval_metric="mlogloss")
        self.model.fit(X_res, y_res - self._label_offset)
        return self

    def predict(self, X):
        return self.model.predict(X) + self._label_offset

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def get_models(profile, n_cat_cols=0, random_state=cfg.RANDOM_STATE, models_to_run=None):
    """Return {name: untrained model}. XGBoost / LightGBM / imbalanced-learn are
    optional: a missing package only skips the models that need it.

    Resource settings (profile "conservative"): on 4.4 M training rows an
    unbounded RandomForest exhausts memory, so max_depth and max_samples bound
    each tree; n_jobs=1 because joblib parallelism deadlocked silently on the
    Windows/Anaconda machine used (same issue as IterativeImputer in step 01)."""
    P = profile
    models = {"RandomForest": RandomForestClassifier(
        n_estimators=P["rf_estimators"], max_depth=P["rf_depth"],
        max_samples=P["rf_max_samples"], class_weight="balanced",
        n_jobs=P["n_jobs"], verbose=2, random_state=random_state)}

    xgb_tree = dict(n_estimators=P["gb_estimators"], max_depth=P["gb_depth"],
                    learning_rate=0.1, subsample=P["gb_subsample"], colsample_bytree=0.8,
                    tree_method="hist", n_jobs=P["n_jobs"], verbosity=1)
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                                          random_state=random_state, **xgb_tree)
        models["XGBoost-Ordinal"] = OrdinalXGBClassifier(
            classes=[1, 2, 3, 4], random_state=random_state, **xgb_tree)
    except ImportError:
        print("[info] xgboost not installed - XGBoost models skipped (pip install xgboost).")

    try:
        from imblearn.over_sampling import SMOTENC  # noqa: F401
        from xgboost import XGBClassifier  # noqa: F401
        models["XGBoost-SMOTE"] = SMOTEXGBClassifier(
            categorical_features=list(range(n_cat_cols)),
            max_rows_before_smote=P["smote_max_rows"], random_state=random_state,
            **xgb_tree)
    except ImportError:
        print("[info] xgboost and/or imbalanced-learn missing - XGBoost-SMOTE skipped.")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=P["gb_estimators"], max_depth=-1, learning_rate=0.1,
            subsample=P["gb_subsample"], colsample_bytree=0.8, subsample_freq=1,
            class_weight="balanced", n_jobs=P["n_jobs"], verbose=1,
            random_state=random_state)
    except ImportError:
        print("[info] lightgbm not installed - model skipped (pip install lightgbm).")

    if models_to_run is not None:
        models = {k: v for k, v in models.items() if k in models_to_run}
    return models


def sample_weights_for_xgboost(y):
    """XGBoost has no native class_weight='balanced': equivalent sample weights."""
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    n, k = len(y), len(classes)
    return y.map(lambda c: n / (k * freq[c])).values


def fit_with_memory_retry(fit_fn, max_retries=3):
    """Run fit_fn(frac) with frac = fraction of training rows (1.0 first). On a
    MemoryError the training subsample is halved and the fit retried, rather
    than crashing - at the price of a model trained on less data, flagged in
    the logs."""
    frac = 1.0
    for attempt in range(max_retries):
        try:
            return fit_fn(frac)
        except MemoryError as e:
            frac /= 2
            print(f"[MemoryError] Training failed ({e}). Retrying with {frac * 100:.1f} % "
                  f"of the rows ({attempt + 1}/{max_retries}).")
    raise MemoryError(f"Training failed after {max_retries} attempts (last fraction: "
                      f"{frac * 100:.1f} %). Reduce n_estimators/max_depth or use more RAM.")


def fit_predict(name, model, X_train, y_train, X_test, random_state=cfg.RANDOM_STATE):
    """Train one model and return its test predictions (labels 1..4)."""
    def do_fit(frac):
        if frac < 1.0:
            X_sub = X_train.sample(frac=frac, random_state=random_state)
            y_sub = y_train.loc[X_sub.index]
        else:
            X_sub, y_sub = X_train, y_train
        if name == "XGBoost":
            # XGBoost expects classes 0..k-1 and has no class_weight option.
            w = sample_weights_for_xgboost(y_sub)
            model.fit(X_sub, y_sub - y_train.min(), sample_weight=w)
            return model.predict(X_test) + y_train.min()
        model.fit(X_sub, y_sub)
        return model.predict(X_test)

    return np.asarray(fit_with_memory_retry(do_fit)).astype(int)


# ------------------------------------------------------------------
# Metrics (computed from a confusion matrix, hence fast enough to bootstrap)
# ------------------------------------------------------------------
_W_QUAD = (np.subtract.outer(np.arange(N_CLASSES), np.arange(N_CLASSES)) ** 2).astype(float)
_W_ABS = np.abs(np.subtract.outer(np.arange(N_CLASSES), np.arange(N_CLASSES))).astype(float)


def cm_from_labels(y_true, y_pred):
    idx = (np.asarray(y_true) - 1) * N_CLASSES + (np.asarray(y_pred) - 1)
    return np.bincount(idx, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)


def metrics_from_cm(cm):
    """cm[i, j] = observations of true class i predicted as j (0-indexed).

    macro_f1      unweighted mean of per-class F1 (main metric: imbalanced classes)
    ordinal_mae   mean absolute error in severity levels
    qwk           quadratic weighted Cohen's kappa
    *_killed      precision / recall / F1 of class 4 (killed)
    """
    cm = cm.astype(float)
    n = cm.sum()
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(axis=0), out=np.zeros(N_CLASSES), where=cm.sum(axis=0) > 0)
    rec = np.divide(tp, cm.sum(axis=1), out=np.zeros(N_CLASSES), where=cm.sum(axis=1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(N_CLASSES), where=(prec + rec) > 0)
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    k = KILLED - 1
    return {
        "macro_f1": f1.mean(),
        "weighted_f1": (f1 * cm.sum(axis=1)).sum() / n,
        "accuracy": tp.sum() / n,
        "ordinal_mae": (_W_ABS * cm).sum() / n,
        "qwk": 1.0 - (_W_QUAD * cm).sum() / (_W_QUAD * expected).sum(),
        "precision_killed": prec[k],
        "recall_killed": rec[k],
        "f1_killed": f1[k],
    }


def save_confusion_matrix(y_test, y_pred, title, path):
    labels = list(range(1, N_CLASSES + 1))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ------------------------------------------------------------------
# SHAP (optional)
# ------------------------------------------------------------------
def shap_analysis(model, X_test, feature_names, variant, model_name, output_dir,
                  sample_size=cfg.SHAP_SAMPLE_SIZE, random_state=cfg.RANDOM_STATE):
    try:
        import shap
    except ImportError:
        print("[info] shap not installed - interpretability skipped (pip install shap).")
        return

    if isinstance(model, OrdinalXGBClassifier):
        # The ordinal model is a bundle of binary trees, which TreeExplainer
        # cannot read directly: explain the median-threshold classifier
        # (P(severity > 2): serious vs non-serious), the most informative one.
        thr = model.thresholds[len(model.thresholds) // 2]
        model = model.models[thr]
        model_name = f"{model_name}_threshold{thr}"
        print(f"[info] XGBoost-Ordinal: SHAP on the binary classifier of threshold "
              f"severity > {thr}.")
    elif isinstance(model, SMOTEXGBClassifier):
        model = model.model     # TreeExplainer needs the inner XGBClassifier

    print(f"\n--- SHAP for {model_name} ({variant}) ---")
    sample = X_test.sample(min(sample_size, len(X_test)), random_state=random_state)
    try:
        shap_values = shap.TreeExplainer(model).shap_values(sample)
    except Exception as e:  # noqa: BLE001
        print(f"[info] SHAP skipped for {model_name} ({variant}): {e}")
        return

    plt.figure()
    # Multiclass outputs: a list of (n_samples, n_features) arrays (old API) or a
    # 3-D array (new API). Handling the 3-D case avoids summary_plot reading it
    # as pairwise INTERACTION values. Average |SHAP| over the class axis.
    if isinstance(shap_values, list):
        values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        n_features = sample.shape[1]
        class_axis = [ax for ax, size in enumerate(shap_values.shape)
                      if ax != 0 and size != n_features]
        values = np.abs(shap_values).mean(axis=class_axis[0] if class_axis else 2)
    else:
        values = shap_values        # binary classifier: signed values (beeswarm)
    shap.summary_plot(values, sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    path = output_dir / f"shap_summary_{variant}_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
