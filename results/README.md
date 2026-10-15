# Results files

Final results of the run on the corrected dataset (2,887,353 rows). `extended_ceinture` = `extended_belt` in the paper.

* `prediction/<feature_set>/` — `results_comparison.csv` (all models, both variants, with ordinal MAE, kappa and killed-class metrics), `results_variant_*.csv`, `per_class_metrics_*.csv`. Feature sets: `base`, `extended`, `extended_ceinture` (final).
* `robustness/` — 5-seed results and bootstrap (geo variant): `robustness_per_seed.csv`, `robustness_summary.csv`, `robustness_bootstrap_ci.csv`, `robustness_bootstrap_diff.csv`.
* `risk_factors/` — correlations, PCA loadings, association rules about serious accidents (the full rule sets exceed 10 MB and are not committed).
* `spatial/` — raw accident counts per municipality. `overview/` — raw-file inventory and missing values.
* `run_all.log`, `run_extra.log` — console logs of the runs.
