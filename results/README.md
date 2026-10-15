# Results files

* `prediction/extended_ceinture/` — **final results** (5 models × 2 variants, feature set `extended_ceinture`):
  `results_comparison.csv`, `results_variant_full.csv`, `results_variant_geo.csv`
  (columns: variant, model, macro_f1, weighted_f1, accuracy).
* `prediction/base/` — only the **last** two models run with the `base` feature set (XGBoost-Ordinal v2 and XGBoost-SMOTE).
  The base results of Random Forest, XGBoost and LightGBM are listed in `EXPERIMENTS.md` (E1).
* `prediction/extended/` — the feature-set test on XGBoost / geo with the five extra equipment variables (E4).
* `robustness/` — five-seed results and bootstrap (E6): `robustness_per_seed.csv`, `robustness_summary.csv`
  (`*_mean`, `*_std`), `robustness_bootstrap_ci.csv`, `robustness_bootstrap_diff.csv`.

Scripts 04 and 05 write richer files when re-run (ordinal MAE, kappa, killed-class P/R/F1 and per-class tables)
into `outputs/04_severity_prediction/<feature_set>/`.
