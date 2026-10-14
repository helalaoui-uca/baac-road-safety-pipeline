"""
Step 02 - Exploratory analysis and risk-factor identification.

Four analyses on the cleaned dataset produced by step 01:
    1. Correlations (Pearson / Spearman) between continuous variables and severity
    2. PCA on the continuous variables
    3. Association rules (Apriori and FP-Growth) between circumstance factors and
       serious accidents (hospitalised or killed)
    4. Heat calendars (hour x weekday and month x weekday)

Figures are written in English to outputs/02_risk_factors/.

Usage:
    python scripts/02_risk_factors.py                 # everything
    python scripts/02_risk_factors.py --skip-rules    # skip the costly rule mining
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config as cfg
from baac.common import encode_severity, load_master, log

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ENGLISH_VAR_NAMES = {
    "age": "age", "heure_accident": "hour_of_day", "vma": "speed_limit",
    "nbv": "num_lanes", "lartpc": "median_width", "larrout": "roadway_width",
    "lat": "latitude", "long": "longitude", "jour_semaine": "day_of_week",
    cfg.TARGET_COL: "severity",
}
SEVERITY_LEGEND = {1: "1 - Unharmed", 2: "2 - Light injury",
                   3: "3 - Hospitalised injury", 4: "4 - Killed"}


# ------------------------------------------------------------------
# 1. Correlations
# ------------------------------------------------------------------
def correlation_analysis(df, out):
    cols = [c for c in cfg.NUMERIC_COLS if c in df.columns] + [cfg.TARGET_COL]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    pearson, spearman = sub.corr(method="pearson"), sub.corr(method="spearman")
    pearson.to_csv(out / "correlation_pearson.csv")
    spearman.to_csv(out / "correlation_spearman.csv")

    print("\nSpearman correlation with severity:")
    print(spearman[cfg.TARGET_COL].drop(cfg.TARGET_COL)
          .sort_values(key=abs, ascending=False).round(3))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, corr, title in ((axes[0], pearson, "Pearson correlation"),
                            (axes[1], spearman, "Spearman correlation")):
        sns.heatmap(corr.rename(index=ENGLISH_VAR_NAMES, columns=ENGLISH_VAR_NAMES),
                    annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out / "correlation_heatmaps.png", dpi=150)
    plt.close()


# ------------------------------------------------------------------
# 2. PCA
# ------------------------------------------------------------------
def pca_analysis(df, out, n_components=5):
    cols = [c for c in cfg.NUMERIC_COLS if c in df.columns]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    n_before = len(sub)
    sub = sub.dropna()      # PCA tolerates no NaN; step 01 normally imputed them
    if len(sub) < n_before:
        print(f"[PCA] {n_before - len(sub):,} rows with residual NaN dropped.")

    scaled = StandardScaler().fit_transform(sub)
    n_components = min(n_components, scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=cfg.RANDOM_STATE)
    scores = pca.fit_transform(scaled)
    explained = pca.explained_variance_ratio_
    print("\nExplained variance per component:")
    for i, v in enumerate(explained, 1):
        print(f"  PC{i}: {v * 100:.1f} % (cumulative: {explained[:i].sum() * 100:.1f} %)")

    pd.DataFrame(pca.components_.T, index=cols,
                 columns=[f"PC{i + 1}" for i in range(n_components)]
                 ).to_csv(out / "pca_loadings.csv")

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, n_components + 1), np.cumsum(explained) * 100, marker="o")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance (%)")
    plt.title("Scree plot - PCA")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "pca_scree_plot.png", dpi=150)
    plt.close()

    plot_df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1],
                            "severity": df.loc[sub.index, cfg.TARGET_COL].values})
    sample = plot_df.sample(min(20_000, len(plot_df)), random_state=cfg.RANDOM_STATE)
    sample["severity_label"] = sample["severity"].map(SEVERITY_LEGEND)
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=sample, x="PC1", y="PC2", hue="severity_label",
                    hue_order=[SEVERITY_LEGEND[k] for k in sorted(SEVERITY_LEGEND)],
                    palette="viridis", alpha=0.4, s=10)
    plt.title("PCA projection (PC1-PC2), coloured by severity")
    plt.legend(title="Severity")
    plt.tight_layout()
    plt.savefig(out / "pca_projection.png", dpi=150)
    plt.close()


# ------------------------------------------------------------------
# 3. Association rules (Apriori and FP-Growth)
# ------------------------------------------------------------------
def _onehot_transactions(df, cols, sample_size):
    """One-hot "variable=value" transaction matrix shared by both algorithms."""
    from mlxtend.preprocessing import TransactionEncoder
    sub = df[cols].copy()
    if sample_size is not None and len(sub) > sample_size:
        sub = sub.sample(sample_size, random_state=cfg.RANDOM_STATE)
        print(f"[Association] Random sample of {sample_size:,} rows (of {len(df):,}).")
    transactions = [[f"{c}={row[c]}" for c in cols
                     if pd.notna(row[c]) and str(row[c]) != cfg.MISSING_LABEL]
                    for _, row in sub.iterrows()]
    # Dense (not sparse) matrix on purpose: avoids an mlxtend/scipy integer
    # overflow bug ("negative dimensions are not allowed") in sparse indexing.
    te = TransactionEncoder()
    return pd.DataFrame(te.fit(transactions).transform(transactions, sparse=False),
                        columns=te.columns_)


def _rules_from_itemsets(itemsets, min_lift, name, out, top_n):
    from mlxtend.frequent_patterns import association_rules
    if itemsets.empty:
        print(f"[!] {name}: no frequent itemset - lower the support threshold.")
        return None
    rules = association_rules(itemsets, metric="lift", min_threshold=min_lift)
    # Keep rules whose consequent concerns serious accidents: they answer the
    # risk-factor question directly.
    severity = rules[rules["consequents"].apply(
        lambda s: "accident_grave=True" in s)].sort_values("lift", ascending=False)
    slug = name.lower().replace("-", "")
    rules.to_csv(out / f"association_rules_{slug}_all.csv", index=False)
    severity.to_csv(out / f"association_rules_{slug}_severity.csv", index=False)
    print(f"\n[{name}] {len(rules)} rules, {len(severity)} about serious accidents.")
    if not severity.empty:
        print(severity[["antecedents", "consequents", "support", "confidence", "lift"]]
              .head(top_n).to_string(index=False))
    return severity


def association_rules_analysis(df, out, top_n=30):
    try:
        from mlxtend.frequent_patterns import apriori, fpgrowth
    except ImportError as e:
        print(f"[!] mlxtend not installed ({e}) - rule mining skipped.")
        return
    cols = [c for c in cfg.CATEGORICAL_FACTORS if c in df.columns] + ["accident_grave"]
    onehot = _onehot_transactions(df, cols, cfg.ASSOC_SAMPLE_SIZE)

    # Neither algorithm is bounded in itemset size by default, which explodes
    # combinatorially on ~190 one-hot columns. Apriori (explicit candidate
    # generation) saturates memory -> higher support, low_memory, capped length.
    # FP-Growth (compressed prefix tree) does not saturate memory but can run
    # forever on deep itemsets -> capped length, more permissive support.
    log("[Association] Apriori (stricter support)...")
    ap = apriori(onehot, min_support=cfg.APRIORI_MIN_SUPPORT, use_colnames=True,
                 max_len=cfg.ASSOC_MAX_LEN, low_memory=True)
    _rules_from_itemsets(ap, cfg.ASSOC_MIN_LIFT, "Apriori", out, top_n)

    log("[Association] FP-Growth (capped itemset size)...")
    fp = fpgrowth(onehot, min_support=cfg.FPGROWTH_MIN_SUPPORT, use_colnames=True,
                  max_len=cfg.ASSOC_MAX_LEN)
    _rules_from_itemsets(fp, cfg.ASSOC_MIN_LIFT, "FP-Growth", out, top_n)
    print(f"\nFrequent itemsets - Apriori: {len(ap):,}; FP-Growth: {len(fp):,}.")


# ------------------------------------------------------------------
# 4. Heat calendars
# ------------------------------------------------------------------
def heat_calendars(df, out):
    month = pd.to_numeric(df.get("mois"), errors="coerce")
    weekday = pd.to_numeric(df.get("jour_semaine"), errors="coerce")
    hour = pd.to_numeric(df.get("heure_accident"), errors="coerce")

    for rows, ylabel, title, fname, size in (
            (month, "Month", "Accidents by month and day of week",
             "heat_calendar_month_weekday.png", (9, 6)),
            (hour, "Hour of day", "Accidents by hour and day of week",
             "heat_calendar_hour_weekday.png", (9, 8))):
        pivot = pd.crosstab(rows, weekday)
        pivot.columns = [DAYS[int(c)] for c in pivot.columns if 0 <= c <= 6]
        plt.figure(figsize=size)
        sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3,
                    cbar_kws={"label": "Number of accidents"})
        plt.title(f"Heat calendar - {title}")
        plt.xlabel("Day of week")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out / fname, dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--skip-rules", action="store_true",
                    help="skip association-rule mining (the costly step)")
    args = ap.parse_args()

    out = cfg.RISK_FACTORS_DIR
    out.mkdir(parents=True, exist_ok=True)

    cols = (set(cfg.NUMERIC_COLS) | set(cfg.CATEGORICAL_FACTORS)
            | {"grav", "mois", "jour_semaine", "heure_accident"})
    master = encode_severity(load_master(columns=sorted(cols)), add_labels=True)

    log("=== 1. Correlations ===")
    correlation_analysis(master, out)
    log("=== 2. PCA ===")
    pca_analysis(master, out)
    if not args.skip_rules:
        log("=== 3. Association rules ===")
        association_rules_analysis(master, out)
    log("=== 4. Heat calendars ===")
    heat_calendars(master, out)
    log(f"Done. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
