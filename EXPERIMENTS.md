# Experiments log

Every experiment of the study, why it was run, what it showed and the decision it led to.
**All numbers come from the final run on the corrected dataset** (2,887,353 user records, 2005–2024), stored in `results/`
(console logs: `results/run_all.log`, `results/run_extra.log`). Prediction results are on the **test years 2022–2024**
after training on 2005–2021. `extended_ceinture` is called `extended_belt` in the paper.

> **History.** A first version of the pipeline joined users to vehicles on `num_acc` only before 2019 and kept
> duplicated `lieux` rows in 2023–2024, inflating the data to 4.88 M rows. Its results were fully recomputed and are obsolete;
> see D11 below and the README ("Vehicle join").

## 1. Data preparation and quality experiments

**Dataset.** 20 yearly vintages (2005–2024) × 4 files (characteristics, places, vehicles, users) joined at user level → **2,887,353 rows**.

| # | Problem observed | Experiment / diagnosis | Decision |
|---|---|---|---|
| D1 | Files differ across years (UTF-8 vs Latin-1, several separators), malformed lines | Three-level reader: strict parse, tolerant parse, separator sniffing | `robust_read_csv`, each fallback logged |
| D2 | 2021/2022 files not found | Typo `carcteristiques`; `Accident_Id` instead of `Num_Acc` in 2022 | Extra keyword and column alias |
| D3 | Safety equipment: single `secu` before 2019, `secu1–3` after | Rebuild boolean `equip_*` indicators | Only the **seatbelt** is reliable before 2019; the five other equipments are structurally False before 2019 |
| D4 | Vehicle categories unused since 2006 | Regularised to 99 | Implemented |
| D5 | Missing codes (empty, `.`, `-1`, sometimes `0`) | `0` is valid for several variables | Column-by-column handling |
| D6 | Absurd GPS values (raw integers "degrees × 100,000" in 2010–2018, decimal commas elsewhere) | Magnitude test per year | Values > 1000 divided by 100,000; sentinel 0 → NaN; coordinates never imputed |
| D7 | Imputation mixed coordinate scales | `lat`/`long` excluded from `IterativeImputer` | Kept. The imputer does not converge in 3 iterations (warning in the log): stated as a limitation |
| D8 | `IterativeImputer` hung on Windows/Jupyter | `n_jobs=1`, 30 trees, depth 8, `max_samples=5 %`, 3 iterations | Kept |
| D9 | `com` is unique only within a department | INSEE code rebuilt from `dep` + `com` with special cases | `build_insee_code` (step 03) |
| D10 | Coordinates outside France | One bounding box per territory (11) | 9,816 rows rejected |
| D11 | Row count dropped abruptly in 2019 | **Confirmed bug.** Pre-2019 users were joined to vehicles on `num_acc` only (2016: 257,286 rows vs 133,422 users). Also `lieux` has several rows per accident in 2023–2024, and a re-run appended to the existing parquet folder | Join on `num_acc` + `num_veh` (from 2019: `id_vehicule`); `drop_duplicates("num_acc")` on `carac`/`lieux` (the first `lieux` row is kept, an arbitrary choice); parquet folder wiped before export; assertions on row counts (`n_rows.txt`) |

Anomaly to investigate: **no valid coordinate at all in 2009.**

## 2. Risk-factor analysis (step 02)

* **Correlations (Spearman with severity).** All weak: age −0.076, `nbv` −0.061, `lat` −0.049, `lartpc` −0.047, weekday +0.040, `larrout` +0.026, `vma` +0.023, hour −0.017, `long` +0.005.
* **PCA** (9 continuous variables; 1,396,160 complete rows): PC1 17.7 %, PC2 13.1 % (30.9 % cumulative), PC3 11.9 % (42.7 %), PC4 11.2 % (53.9 %), PC5 11.1 % (65.0 %). No component aligns with severity.
* **Association rules** (300,000-row sample, lift ≥ 1.2, itemsets ≤ 4). Apriori (support ≥ 5 %): 47,716 rules, 1,641 about serious accidents; best lift ≈ 3.0
  (bidirectional traffic + built-up area=1 → department road (`catr`=3) + serious accident; support 6.9 %, confidence 33.5 %).
  FP-Growth (support ≥ 2 %): 172,932 rules, 6,270 about serious accidents; the highest lifts (≈ 4.4–7.1) involve `situ`=3 (accident on the hard shoulder), `obsm`=0 (no mobile obstacle struck) and collision types `col`=6, 7.
  The rule files covering all consequents (> 60 MB) are not committed; only the "severity" subsets are in `results/risk_factors/`.
* **Heat calendars:** `figures/risk_factors/heat_calendar_*.png`.

## 3. Spatial segmentation (step 03)

Run on **accidents** (de-duplicated on `num_acc`): 653,817 accidents with valid GPS (**none in 2009**), 635,984 of them in mainland France.

| Method | Setting | Result |
|---|---|---|
| DBSCAN | haversine, eps 0.5 km, min_samples 15, sample 150,000 accidents | 383 clusters, 59.5 % noise |
| HDBSCAN | min_cluster_size 25, sample 100,000 | 451 clusters, 40.3 % noise |
| ST-DBSCAN, absolute time | eps 0.5 km and 2 h, min_samples 8, sample 50,000 | 5 clusters, 99.9 % noise |
| ST-DBSCAN, hour of day | same thresholds | 171 clusters, 87.4 % noise |

* Near-total noise in absolute-time ST-DBSCAN is expected (two accidents at the same place on the same day are rare over 20 years).
* Without a population file the commune ranking is a **raw count** (exposure, not intrinsic risk). Top: INSEE 13055 (Marseille) 13,492; 35238 (Rennes) 5,116; 34172 (Montpellier) 4,997.
* DBSCAN labels are the `cluster_dbscan` variable of the "geo" variant (339,326 user rows).

## 4. Severity prediction (steps 04–05)

### 4.0 Design

* **Target** `gravite_ordinale`: 1 unharmed, 2 light, 3 hospitalised, 4 killed.
* **Validation:** temporal split, train 2005–2021, test 2022–2024. `full`: 2,509,655 train / 377,279 test rows. `geo`: 253,596 / 85,730 (users of the DBSCAN-sampled accidents, 339,326 rows).
  Test class counts: full 159,949 / 149,293 / 57,657 / 10,380 (killed 2.8 %); geo 36,348 / 33,714 / 13,232 / 2,436.
* **Metrics:** macro-F1 (main), weighted-F1, accuracy, ordinal MAE, quadratic weighted kappa, precision/recall/F1 of *killed*.
* **Conservative resource profile**, no tuning: `n_jobs=1`; RF 150 trees depth 14, 10 % of rows per tree; boosting 250 trees depth 7, lr 0.1, row subsample 0.2; SMOTENC on ≤ 300,000 rows (rebalanced sets: geo 411,292 rows, full 492,076).
* **Models:** Random Forest, XGBoost, LightGBM (class weights); XGBoost-Ordinal (Frank & Hall, three binary models, square-root-damped rebalancing); XGBoost-SMOTE (SMOTENC).

### E1 — Feature sets (macro-F1, `base` → `extended_ceinture`, and `extended`)

The nine variables added in `extended_ceinture` (`equip_ceinture, catu, place, motor, senc, mois, locp, actp, etatp`) carry the gain. The five extra equipments of `extended`
(`equip_casque, equip_enfant, equip_gilet, equip_airbag_2rm, equip_gants_2rm`) add almost nothing and risk a temporal-coding artefact (coded only from 2019), so they were dropped.

| Model | full | geo |
|---|---|---|
| XGBoost-Ordinal | 0.477 → 0.515 (+0.038); ext. 0.517 | 0.482 → 0.523 (+0.041); ext. 0.521 |
| XGBoost-SMOTE | 0.476 → 0.516 (+0.040); ext. 0.514 | 0.488 → 0.524 (+0.036); ext. 0.524 |
| XGBoost | 0.458 → 0.495 (+0.037); ext. 0.497 | 0.482 → 0.516 (+0.034); ext. 0.518 |
| LightGBM | 0.451 → 0.489 (+0.038); ext. 0.491 | 0.476 → 0.513 (+0.036); ext. 0.516 |
| RandomForest | 0.438 → 0.466 (+0.028); ext. 0.468 | 0.463 → 0.495 (+0.032); ext. 0.498 |

*Finding:* the gain is homogeneous across models → the variables matter more than the model.

### E2 — Final comparison, `extended_ceinture`

**geo**

| Model | Macro-F1 | Weighted-F1 | Accuracy | Ordinal MAE | Weighted kappa | killed P / R | killed F1 |
|---|---|---|---|---|---|---|---|
| XGBoost-SMOTE | 0.524 | 0.658 | 0.661 | 0.395 | 0.593 | 0.25 / 0.19 | 0.22 |
| XGBoost-Ordinal | 0.523 | 0.643 | 0.646 | 0.424 | 0.594 | 0.21 / 0.37 | 0.27 |
| XGBoost | 0.516 | 0.645 | 0.635 | 0.450 | 0.583 | 0.17 / 0.52 | 0.25 |
| LightGBM | 0.513 | 0.642 | 0.630 | 0.463 | 0.576 | 0.16 / 0.56 | 0.25 |
| RandomForest | 0.495 | 0.625 | 0.615 | 0.496 | 0.540 | 0.15 / 0.58 | 0.24 |

**full**

| Model | Macro-F1 | Weighted-F1 | Accuracy | Ordinal MAE | Weighted kappa | killed P / R | killed F1 |
|---|---|---|---|---|---|---|---|
| XGBoost-SMOTE | 0.516 | 0.653 | 0.660 | 0.397 | 0.580 | 0.24 / 0.20 | 0.22 |
| XGBoost-Ordinal | 0.515 | 0.639 | 0.644 | 0.433 | 0.594 | 0.20 / 0.53 | 0.29 |
| XGBoost | 0.495 | 0.634 | 0.621 | 0.490 | 0.564 | 0.14 / 0.70 | 0.24 |
| LightGBM | 0.489 | 0.630 | 0.614 | 0.505 | 0.555 | 0.14 / 0.72 | 0.23 |
| RandomForest | 0.466 | 0.611 | 0.599 | 0.536 | 0.522 | 0.13 / 0.71 | 0.22 |

Per-class F1 (geo): unharmed 0.76–0.78, light 0.61–0.64, hospitalised 0.37–0.46, killed 0.22–0.27 (`results/prediction/extended_ceinture/per_class_metrics_*.csv`).
*Findings:* the difficulty lies in the two serious classes. XGBoost-SMOTE and XGBoost-Ordinal tie on macro-F1; SMOTE is best on accuracy, weighted-F1 and MAE, Ordinal on kappa and on *killed* recall/F1.
RF, LightGBM and XGBoost flag most killed users (recall 0.52–0.58, geo) but with precision 0.15–0.17.

### E3 — Robustness: 5 seeds and paired bootstrap (geo, `results/robustness/`)

| Model | Macro-F1 | Ordinal MAE | Weighted kappa | F1 killed | Recall killed |
|---|---|---|---|---|---|
| XGBoost-Ordinal | 0.5205 ± 0.0029 | 0.4258 ± 0.0015 | 0.5956 ± 0.0011 | 0.2672 ± 0.0075 | 0.376 ± 0.009 |
| XGBoost-SMOTE | 0.5242 ± 0.0006 | 0.3944 ± 0.0004 | 0.5946 ± 0.0008 | 0.2156 ± 0.0021 | 0.191 ± 0.003 |
| XGBoost | 0.5179 ± 0.0011 | 0.4481 ± 0.0016 | 0.5851 ± 0.0014 | 0.2551 ± 0.0013 | 0.511 ± 0.010 |

Paired bootstrap (1,000 resamples, A − B, 95 % CI): Ordinal − SMOTE macro-F1 −0.0007 [−0.0051; 0.0042] (p = 0.79, **tie**), kappa +0.0007 (tie), F1 killed +0.056 [0.041; 0.072], MAE +0.029 (worse).
Both beat XGBoost on macro-F1 (Ordinal +0.0066, SMOTE +0.0073, CI excluding 0). Limit: users of one accident are correlated, so the intervals are probably slightly too narrow.

### E4 — Interpretability (SHAP, 5,000 test rows)

Figures in `figures/prediction/extended_ceinture/`. The Ordinal plots show the **signed** SHAP values of the binary classifier P(severity > light injury); the SMOTE plots show the unsigned mean |SHAP| of the multiclass model.
In the geo variant the spatial cluster (`cluster_dbscan`) ranks second, behind the seatbelt. For integer-coded categorical variables the colour scale has no meaning: read the spread, not the colour.

## 5. Decisions, limitations and open points

**Retained configuration:** feature set `extended_ceinture` (`extended_belt`); XGBoost-Ordinal and XGBoost-SMOTE presented as equivalent on macro-F1 (Ordinal if missing a fatal accident is costlier than a false alarm, SMOTE for the best overall error).

**Limitations**

1. Explanatory, not predictive: several variables are known only after the accident.
2. *Killed* class: precision about 0.21–0.25 at best, not operationally usable.
3. `full` and `geo` have different test sets.
4. No hyper-parameter tuning; robustness protocol only for the geo variant and three models.
5. The bootstrap ignores the correlation between users of the same accident.
6. First `lieux` row kept per accident in 2023–2024 (arbitrary); imputation not converged after 3 iterations; no valid GPS in 2009.
7. Spatial rankings are raw counts unless a population file is provided.

**Next steps:** investigate 2009 coordinates; tune hyper-parameters (`powerful` profile in `config.py`); robustness on the full variant; probability calibration and operating thresholds for *killed*.
