"""
Step 00 - Preliminary statistical study of the raw BAAC files (2005-2024).

Part 1 - dataset overview
    * inventory of found / missing files
    * rows and columns per file and per year
    * column presence by year, to spot nomenclature breaks (e.g. `secu` ->
      `secu1/secu2/secu3` in 2019, appearance of vma/lat/long)

Part 2 - missing values
    * BAAC "sentinel" codes (empty cell, ".", -1) normalised to real NaN
    * missing-rate table (% per column x year) and heatmap per file type

Expected input layout (the matching is tolerant to file naming)::

    data/raw/<year>/caracteristiques-<year>.csv   (also lieux, vehicules, usagers)

Usage:
    python scripts/00_data_overview.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config as cfg
from baac.common import log

FILE_TYPES = cfg.FILE_KEYWORDS
# "0" is a VALID code for some variables, so it is not treated as missing here;
# list columns in ZERO_AS_NA_COLUMNS only after checking the BAAC codebook.
ZERO_AS_NA_COLUMNS = []


def find_file(year, keywords):
    year_dir = cfg.RAW_DIR / str(year)
    base = year_dir if year_dir.exists() else cfg.RAW_DIR
    cands = []
    for kw in keywords:
        cands += list(base.glob(f"*{kw}*{year}*.csv")) + list(base.glob(f"*{kw}*.csv"))
    cands = sorted(set(cands))
    if year_dir.exists():
        return cands[0] if cands else None
    cands = [c for c in cands if str(year) in c.name]
    return cands[0] if cands else None


def robust_read_csv(path):
    """Encoding (utf-8 / latin-1) and separator (, ; tab |) change across years."""
    last_err = None
    for enc in ("utf-8", "latin-1"):
        for sep in (",", ";", "\t", "|"):
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                if df.shape[1] > 1:
                    return df, enc, sep
            except Exception as e:  # noqa: BLE001
                last_err = e
    raise ValueError(f"Cannot read {path}: {last_err}")


def normalize_missing(df):
    df = df.replace(cfg.SENTINELS_GENERIC, pd.NA)
    for col in ZERO_AS_NA_COLUMNS:
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA)
    return df


def build_inventory():
    inventory, frames = [], {ft: {} for ft in FILE_TYPES}
    for year in cfg.YEARS:
        for ftype, kws in FILE_TYPES.items():
            path = find_file(year, kws)
            if path is None:
                inventory.append({"year": year, "file": ftype, "found": False})
                continue
            df, enc, sep = robust_read_csv(path)
            frames[ftype][year] = df
            inventory.append({"year": year, "file": ftype, "found": True,
                              "n_rows": len(df), "n_columns": df.shape[1],
                              "encoding": enc, "separator": sep, "path": str(path)})
    return pd.DataFrame(inventory), frames


def column_presence(frames, ftype):
    presence = {}
    for year, df in frames[ftype].items():
        for col in df.columns:
            presence.setdefault(col, {})[year] = True
    return pd.DataFrame(presence).T.reindex(columns=list(cfg.YEARS)).notna()


def missing_summary(frames):
    rows = []
    for ftype, years in frames.items():
        for year, df in years.items():
            rates = normalize_missing(df).isna().mean() * 100
            for col, rate in rates.items():
                rows.append({"file": ftype, "year": year, "column": col,
                             "pct_missing": round(float(rate), 2)})
    return pd.DataFrame(rows)


def plot_missing_heatmap(missing_df, ftype, out_path):
    subset = missing_df[missing_df.file == ftype]
    if subset.empty:
        return
    pivot = subset.pivot(index="column", columns="year", values="pct_missing")
    plt.figure(figsize=(min(20, 1 + 0.6 * pivot.shape[1]), max(4, 0.35 * pivot.shape[0])))
    sns.heatmap(pivot, cmap="Reds", cbar_kws={"label": "% missing"}, linewidths=0.3)
    plt.title(f"Missing-value rate by variable and year - {ftype.upper()} file")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    out = cfg.OVERVIEW_DIR
    out.mkdir(parents=True, exist_ok=True)

    log("=== Part 1: dataset inventory ===")
    inv, frames = build_inventory()
    inv.to_csv(out / "inventory.csv", index=False)
    print(inv[["year", "file", "found", "n_rows", "n_columns"]].to_string(index=False))
    missing_files = inv[~inv.found]
    if not missing_files.empty:
        print(f"\n[!] {len(missing_files)} file(s) not found - check data/raw/ and naming:")
        print(missing_files[["year", "file"]].to_string(index=False))
    print("\nTotal rows per file type:")
    print(inv.groupby("file")["n_rows"].sum())

    for ftype in FILE_TYPES:
        if not frames[ftype]:
            continue
        presence = column_presence(frames, ftype)
        presence.to_csv(out / f"columns_{ftype}.csv")
        inconsistent = presence[~presence.all(axis=1)]
        if not inconsistent.empty:
            print(f"\n[{ftype}] columns absent in some years (potential breaks):")
            print(list(inconsistent.index))

    log("=== Part 2: missing values ===")
    miss = missing_summary(frames)
    miss.to_csv(out / "missing_values.csv", index=False)
    for ftype in FILE_TYPES:
        plot_missing_heatmap(miss, ftype, out / f"missing_heatmap_{ftype}.png")

    log(f"Done. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
