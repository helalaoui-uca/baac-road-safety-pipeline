"""
Step 05 - Robustness of the prediction results.

1. Seed variability: each model is retrained with several random seeds ->
   mean +/- standard deviation of every metric.
2. Metrics suited to an ORDERED target: ordinal MAE (in severity levels) and
   quadratic weighted kappa, plus precision / recall / F1 of the "killed" class.
3. Bootstrap on the test set (95 % confidence intervals) and PAIRED bootstrap of
   the differences between models (same resampled rows for every model), to tell
   whether an advantage is statistically distinguishable from chance.

Note: the bootstrap resamples test rows independently, whereas users of the same
accident are correlated, so the intervals are probably slightly too narrow.

Usage:
    python scripts/05_robustness.py                        # geo, 3 models, 5 seeds
    python scripts/05_robustness.py --variants geo full --seeds 0 1 2

Runtime ~ (seeds x models) x the training time of one model; XGBoost-SMOTE is the
slowest (~25 min per seed on the "geo" variant with the conservative profile).

Outputs (outputs/04_severity_prediction/<feature_set>/robustness/):
    robustness_per_seed.csv, robustness_summary.csv,
    robustness_bootstrap_ci.csv, robustness_bootstrap_diff.csv
"""

import argparse
import gc
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as cfg
from baac import prediction as pred
from baac.common import log


def bootstrap(y_true, preds, n_boot, seed=12345):
    """Paired bootstrap: the same resampled indices are used for every model."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    out = {m: [] for m in preds}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for m, p in preds.items():
            out[m].append(pred.metrics_from_cm(pred.cm_from_labels(y_true[idx], p[idx])))
        if (b + 1) % 200 == 0:
            log(f"bootstrap {b + 1}/{n_boot}")
    return {m: pd.DataFrame(v) for m, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--feature-set", default=cfg.DEFAULT_FEATURE_SET,
                    choices=sorted(cfg.FEATURE_SETS))
    ap.add_argument("--profile", default=cfg.DEFAULT_PROFILE, choices=sorted(cfg.PROFILES))
    ap.add_argument("--variants", nargs="+", default=["geo"], choices=["full", "geo"])
    ap.add_argument("--models", nargs="+", default=cfg.ROBUSTNESS_MODELS, choices=pred.ALL_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg.ROBUSTNESS_SEEDS)
    ap.add_argument("--reference-seed", type=int, default=cfg.ROBUSTNESS_SEEDS[0],
                    help="seed whose predictions feed the bootstrap")
    ap.add_argument("--n-boot", type=int, default=cfg.BOOTSTRAP_REPLICATES)
    args = ap.parse_args()

    out = cfg.PREDICTION_DIR / args.feature_set / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    profile = cfg.PROFILES[args.profile]
    master = pred.load_prediction_master(args.feature_set)

    rows, ci_rows, diff_rows = [], [], []
    for variant in args.variants:
        log(f"===== Variant {variant} =====")
        X_train, y_train, X_test, y_test, n_cat = pred.prepare_variant(
            master, variant, args.feature_set)
        log(f"train={len(X_train):,} test={len(X_test):,} features={X_train.shape[1]}")
        ref_preds = {}

        for seed in args.seeds:
            models = pred.get_models(profile, n_cat, seed, args.models)
            for name, model in models.items():
                log(f"--- {name} ({variant}) seed {seed} ---")
                y_pred = pred.fit_predict(name, model, X_train, y_train, X_test, seed)
                m = pred.metrics_from_cm(pred.cm_from_labels(y_test, y_pred))
                rows.append({"variant": variant, "model": name, "seed": seed, **m})
                log("   " + " | ".join(f"{k}={v:.3f}" for k, v in m.items()))
                if seed == args.reference_seed:
                    ref_preds[name] = y_pred
                del model
                gc.collect()
            # incremental save: nothing is lost if the run is interrupted
            pd.DataFrame(rows).to_csv(out / "robustness_per_seed.csv", index=False)

        if ref_preds:
            log(f"Bootstrap ({args.n_boot} replicates) - seed {args.reference_seed}")
            boots = bootstrap(y_test, ref_preds, args.n_boot)
            for name, bdf in boots.items():
                for metric in bdf.columns:
                    lo, hi = np.percentile(bdf[metric], [2.5, 97.5])
                    ci_rows.append({"variant": variant, "model": name, "metric": metric,
                                    "estimate": bdf[metric].mean(),
                                    "ci95_low": lo, "ci95_high": hi})
            for a, b in itertools.combinations(boots, 2):
                for metric in boots[a].columns:
                    d = boots[a][metric] - boots[b][metric]
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                    diff_rows.append({"variant": variant, "model_A": a, "model_B": b,
                                      "metric": metric, "diff_A_minus_B": d.mean(),
                                      "ci95_low": lo, "ci95_high": hi, "p_bootstrap": p,
                                      "significant_95": not (lo <= 0 <= hi)})
            pd.DataFrame(ci_rows).to_csv(out / "robustness_bootstrap_ci.csv", index=False)
            pd.DataFrame(diff_rows).to_csv(out / "robustness_bootstrap_diff.csv", index=False)
        del X_train, X_test, y_train, y_test
        gc.collect()

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "robustness_per_seed.csv", index=False)
    metric_cols = [c for c in per_seed.columns if c not in ("variant", "model", "seed")]
    g = per_seed.groupby(["variant", "model"])[metric_cols]
    summary = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).reset_index()
    summary.to_csv(out / "robustness_summary.csv", index=False)

    print("\n=== Mean +/- standard deviation over seeds ===")
    for _, r in summary.iterrows():
        print(f"\n{r['variant']} / {r['model']}")
        for mc in metric_cols:
            print(f"   {mc:17s} {r[mc + '_mean']:.3f} +/- {r[mc + '_std']:.3f}")
    if diff_rows:
        d = pd.DataFrame(diff_rows)
        print("\n=== Paired bootstrap differences ===")
        print(d[d["metric"].isin(["macro_f1", "ordinal_mae", "qwk", "f1_killed"])]
              .round(4).to_string(index=False))
    log(f"Done. Files in {out.resolve()}")


if __name__ == "__main__":
    main()
