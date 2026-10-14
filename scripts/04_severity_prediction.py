"""
Step 04 - Severity prediction (ordinal target: unharmed < light < hospitalised < killed).

Two data variants
    full : all rows, no spatial variable
    geo  : geolocated subset (accidents of the DBSCAN sample) + `cluster_dbscan`

Five models per variant
    RandomForest, XGBoost, LightGBM          class-weighted
    XGBoost-Ordinal                          Frank & Hall decomposition, damped weights
    XGBoost-SMOTE                            SMOTENC on a stratified subsample

Validation is temporal (train 2005-2021, test 2022-2024). The main metric is
the macro-F1; ordinal MAE, quadratic weighted kappa and the precision / recall /
F1 of the "killed" class are also reported. SHAP is computed for the best model
of each variant.

Usage examples:
    python scripts/04_severity_prediction.py                         # all models, both variants
    python scripts/04_severity_prediction.py --variants geo          # geo only (fast check)
    python scripts/04_severity_prediction.py --models XGBoost-Ordinal --variants geo
    python scripts/04_severity_prediction.py --feature-set base      # reproduce the "base" results
    python scripts/04_severity_prediction.py --profile powerful      # workstation with lots of RAM

Results are merged with an existing results CSV in the output folder, so
re-running a single model does not erase the others.
"""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import classification_report

import config as cfg
from baac import prediction as pred
from baac.common import log


def merge_with_previous(new, csv_path, key=("variant", "model")):
    """Replace rows of re-run models, keep the others from the previous run."""
    if new is None or not csv_path.exists():
        return new
    old = pd.read_csv(csv_path)
    merged = old.merge(new[list(key)].drop_duplicates(), on=list(key), how="left",
                       indicator=True)
    old = old[merged["_merge"] == "left_only"]
    return pd.concat([old, new], ignore_index=True)


def run_variant(master, variant, feature_set, profile, models_to_run, out):
    print(f"\n=== Variant '{variant}' ===")
    X_train, y_train, X_test, y_test, n_cat = pred.prepare_variant(master, variant, feature_set)
    models = pred.get_models(profile, n_cat, cfg.RANDOM_STATE, models_to_run)

    rows, per_class, fitted = [], [], {}
    for name, model in models.items():
        log(f"--- Training {name} ({variant}) on {len(X_train):,} rows "
            f"({X_train.shape[1]} features) ---")
        y_pred = pred.fit_predict(name, model, X_train, y_train, X_test)
        log(f"--- {name} ({variant}) done ---")
        fitted[name] = model

        print(classification_report(y_test, y_pred, zero_division=0))
        metrics = pred.metrics_from_cm(pred.cm_from_labels(y_test, y_pred))
        print("  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        rows.append({"variant": variant, "model": name, "feature_set": feature_set, **metrics})

        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        for cls in ("1", "2", "3", "4"):
            per_class.append({"variant": variant, "model": name, "class": int(cls),
                              "precision": report[cls]["precision"],
                              "recall": report[cls]["recall"],
                              "f1": report[cls]["f1-score"],
                              "support": int(report[cls]["support"])})
        pred.save_confusion_matrix(
            y_test, y_pred, f"Confusion matrix - {name} ({variant})",
            out / f"confusion_{variant}_{name}.png")

    results = pd.DataFrame(rows)
    if results.empty:
        return results, pd.DataFrame(per_class)

    best = results.sort_values("macro_f1", ascending=False).iloc[0]["model"]
    pred.shap_analysis(fitted[best], X_test, X_train.columns.tolist(), variant, best, out)
    del X_train, X_test, y_train, y_test, fitted
    gc.collect()
    return results, pd.DataFrame(per_class)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--feature-set", default=cfg.DEFAULT_FEATURE_SET,
                    choices=sorted(cfg.FEATURE_SETS))
    ap.add_argument("--profile", default=cfg.DEFAULT_PROFILE, choices=sorted(cfg.PROFILES))
    ap.add_argument("--variants", nargs="+", default=["full", "geo"], choices=["full", "geo"])
    ap.add_argument("--models", nargs="+", default=None, choices=pred.ALL_MODELS)
    args = ap.parse_args()

    out = cfg.PREDICTION_DIR / args.feature_set
    out.mkdir(parents=True, exist_ok=True)
    profile = cfg.PROFILES[args.profile]
    log(f"Feature set: {args.feature_set} | profile: {args.profile} | output: {out}")

    master = pred.load_prediction_master(args.feature_set)

    for variant in args.variants:
        results, per_class = run_variant(master, variant, args.feature_set, profile,
                                         args.models, out)
        if results.empty:
            continue
        results = merge_with_previous(results, out / f"results_variant_{variant}.csv")
        results.to_csv(out / f"results_variant_{variant}.csv", index=False)
        pc_path = out / f"per_class_metrics_{variant}.csv"
        merge_with_previous(per_class, pc_path, key=("variant", "model")).to_csv(
            pc_path, index=False)

    # Final comparison from the files on disk (includes variants not re-run).
    all_results = []
    for variant in ("full", "geo"):
        p = out / f"results_variant_{variant}.csv"
        if p.exists():
            all_results.append(pd.read_csv(p))
    comparison = pd.concat(all_results, ignore_index=True)
    comparison.to_csv(out / "results_comparison.csv", index=False)
    print("\n=== Comparison ===")
    print(comparison.sort_values(["variant", "macro_f1"], ascending=[True, False])
          .round(4).to_string(index=False))
    log(f"Done. Results, confusion matrices and SHAP figures in {out.resolve()}")


if __name__ == "__main__":
    main()
