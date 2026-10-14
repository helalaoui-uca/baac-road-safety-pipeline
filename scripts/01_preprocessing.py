"""
Step 01 - BAAC data preparation and feature engineering (2005-2024).

Pipeline
    1. Robust reading of the four files of each year (encoding / separator vary).
    2. Harmonisation of known nomenclature breaks:
         - `secu` (single column, before 2019) -> `secu1/secu2/secu3` (from 2019);
           harmonised boolean equipment indicators are rebuilt over the whole period;
         - vehicle category codes unused since 2006 are regularised to "other".
    3. Join of the four files at user level (num_acc, id_vehicule).
    4. Concatenation of the 20 years into a single dataset.
    5. Normalisation of missing-value sentinels (empty, ".", -1) to NaN, column
       by column (no blind replacement).
    6. Feature engineering (hour, user age, day of week, weekend).
    7. Differentiated imputation:
         - continuous / ordinal variables -> IterativeImputer (MissForest-like);
         - nominal categorical variables -> explicit "Non renseigné" category.
    8. Coordinates (lat/long) are cleaned separately and NEVER imputed.

Output: cleaned dataset in data/processed/parquet (hive-partitioned by year) and
data/processed/csv (one file per year).

Usage:
    python scripts/01_preprocessing.py
    python scripts/01_preprocessing.py --years 2021 2022    # quick test on a subset
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

import config as cfg
from baac.common import log

SEP_CANDIDATES = (",", ";", "\t", "|")
# The 2022 "caracteristiques" file uses `Accident_Id` instead of `Num_Acc`.
COLUMN_ALIASES = {"accident_id": "num_acc"}


# ------------------------------------------------------------------
# Robust reading
# ------------------------------------------------------------------
def find_file(year_dir, keywords, year=None):
    """First CSV in `year_dir` matching a keyword; if `year` is given, the file
    name must also contain it (flat folder holding all years)."""
    cands = []
    for kw in keywords:
        cands += list(year_dir.glob(f"*{kw}*.csv")) + list(year_dir.glob(f"*{kw}*.CSV"))
    if year is not None:
        cands = [c for c in cands if str(year) in c.name]
    cands = sorted(set(cands))
    return cands[0] if cands else None


def robust_read_csv(path, debug=False):
    """Three reading levels:
    1. strict parse over (encoding x separator) candidates;
    2. tolerant parse that skips malformed lines (BAAC is raw police data and
       some old years have rows whose field count disagrees with the header);
    3. automatic separator sniffing.
    """
    last_err = None
    for enc in ("utf-8", "latin-1"):
        for sep in SEP_CANDIDATES:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                if df.shape[1] > 1:
                    return df
            except Exception as e:  # noqa: BLE001
                last_err = e
                if debug:
                    print(f"  [debug] {path.name}: enc={enc} sep={sep!r} failed: {e}")
    for enc in ("utf-8", "latin-1"):
        for sep in SEP_CANDIDATES:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python",
                                 on_bad_lines="skip")
                if df.shape[1] > 1:
                    print(f"  [!] {path.name}: tolerant reading used (malformed lines "
                          f"skipped, enc={enc}, sep={sep!r}).")
                    return df
            except Exception as e:  # noqa: BLE001
                last_err = e
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, sep=None, encoding=enc, engine="python",
                             on_bad_lines="skip")
            if df.shape[1] > 1:
                print(f"  [!] {path.name}: separator auto-detected (enc={enc}).")
                return df
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ValueError(f"Cannot read {path}: {last_err}")


def load_year_files(year):
    # Two supported layouts: RAW_DIR/<year>/*.csv  or  a flat RAW_DIR/*_<year>.csv
    year_dir = cfg.RAW_DIR / str(year)
    name_year = None
    if not year_dir.exists():
        year_dir, name_year = cfg.RAW_DIR, year
    if not year_dir.exists():
        print(f"[{year}] folder not found ({year_dir}), year skipped.")
        return None
    dfs = {}
    for ftype, kws in cfg.FILE_KEYWORDS.items():
        path = find_file(year_dir, kws, name_year)
        if path is None:
            print(f"[{year}] file '{ftype}' not found, year skipped.")
            return None
        df = robust_read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns=COLUMN_ALIASES)
        if "num_acc" not in df.columns:
            print(f"[{year}] column 'num_acc' missing in '{ftype}' "
                  f"(columns: {list(df.columns)}), year skipped.")
            return None
        dfs[ftype] = df
    return dfs


# ------------------------------------------------------------------
# Nomenclature harmonisation
# ------------------------------------------------------------------
def harmonize_secu(users):
    """Harmonise safety-equipment coding over 2005-2024.

    Before 2019: one two-digit column `secu` (tens = equipment existence, units =
    use). Only the seatbelt can be rebuilt reliably from it.
    From 2019: up to three columns `secu1..3`, each coding one equipment used
    (1 seatbelt, 2 helmet, 3 child restraint, 4 reflective vest, 5 airbag, 6 gloves).

    Output: boolean indicators equip_ceinture, equip_casque, equip_enfant,
    equip_gilet, equip_airbag_2rm, equip_gants_2rm. Except equip_ceinture, they
    are structurally False before 2019 (information absent, not "not worn").
    """
    df = users.copy()
    equip_map = {1: "equip_ceinture", 2: "equip_casque", 3: "equip_enfant",
                 4: "equip_gilet", 5: "equip_airbag_2rm", 6: "equip_gants_2rm"}
    for col in equip_map.values():
        df[col] = False
    if "secu" in df.columns:
        usage = pd.to_numeric(df["secu"], errors="coerce") % 10
        df["equip_ceinture"] = usage.eq(1)
    secu_cols = [c for c in ("secu1", "secu2", "secu3") if c in df.columns]
    for c in secu_cols:
        code = pd.to_numeric(df[c], errors="coerce")
        for k, name in equip_map.items():
            df[name] = df[name] | code.eq(k)
    return df


def harmonize_catv(vehicles):
    """Regularise vehicle-category codes unused since 2006 to 99 ("other")."""
    df = vehicles.copy()
    if "catv" not in df.columns:
        return df
    obsolete = {4, 5, 6, 8, 9, 11, 12, 18, 19}
    catv = pd.to_numeric(df["catv"], errors="coerce")
    df.loc[catv.isin(obsolete), "catv"] = 99
    return df


# ------------------------------------------------------------------
# Join and concatenation
# ------------------------------------------------------------------
def merge_year(dfs, year):
    carac, lieux = dfs["caracteristiques"], dfs["lieux"]
    veh = harmonize_catv(dfs["vehicules"])
    usa = harmonize_secu(dfs["usagers"])
    keys = ["num_acc"]
    if "id_vehicule" in usa.columns and "id_vehicule" in veh.columns:
        keys.append("id_vehicule")
    elif cfg.JOIN_ON_NUM_VEH and "num_veh" in usa.columns and "num_veh" in veh.columns:
        keys.append("num_veh")          # pre-2019 vehicle key
    n_users = len(usa)
    if len(keys) == 2:
        veh = veh.drop_duplicates(subset=keys)     # one row per vehicle
    merged = usa.merge(veh, on=keys, how="left", suffixes=("", "_veh"))
    if len(merged) > n_users:
        print(f"  [WARNING {year}] user<->vehicle join on {keys} fans out: "
              f"{len(merged):,} rows for {n_users:,} users "
              f"(x{len(merged) / n_users:.2f}). Each user is paired with several "
              f"vehicles; see README 'Known issue' / config.JOIN_ON_NUM_VEH.")
    # Accident-level tables must have ONE row per accident. The 2023 and 2024 `lieux`
    # files hold several rows for some accidents (about 16,000 extra rows per year);
    # joining them as-is would duplicate users. The first row of each accident is kept.
    for name, tab in (("caracteristiques", carac), ("lieux", lieux)):
        n_dup = int(tab.duplicated("num_acc").sum())
        if n_dup:
            print(f"  [{year}] '{name}': {n_dup:,} extra rows for already-listed accidents "
                  f"dropped (first row kept).")
    carac = carac.drop_duplicates("num_acc", keep="first")
    lieux = lieux.drop_duplicates("num_acc", keep="first")
    merged = merged.merge(carac, on="num_acc", how="left", suffixes=("", "_carac"))
    merged = merged.merge(lieux, on="num_acc", how="left", suffixes=("", "_lieux"))
    assert len(merged) == n_users, (f"[{year}] row count changed by the join: "
                                    f"{n_users:,} users -> {len(merged):,} rows")
    merged["annee_dataset"] = year
    return merged


def build_master_dataset(years):
    frames = []
    for year in years:
        dfs = load_year_files(year)
        if dfs is None:
            continue
        merged = merge_year(dfs, year)
        frames.append(merged)
        print(f"[{year}] {len(merged):,} user rows after join.")
    if not frames:
        raise RuntimeError("No year could be loaded - check data/raw/.")
    master = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\nConsolidated dataset: {len(master):,} rows, {master.shape[1]} columns, "
          f"{len(frames)} years.")
    return master


# ------------------------------------------------------------------
# Missing values and feature engineering
# ------------------------------------------------------------------
def normalize_missing(df):
    """Convert BAAC sentinels to NaN column by column, excluding columns where
    0/False is legitimate (equip_* indicators)."""
    df = df.copy()
    protected = [c for c in df.columns if c.startswith("equip_")] + ["annee_dataset"]
    for col in df.columns:
        if col not in protected:
            df[col] = df[col].replace(cfg.SENTINELS_GENERIC, np.nan)
    return df


def _full_year(an):
    an = pd.to_numeric(an, errors="coerce")
    return np.where(an < 100, an + 2000, an)   # 2-digit years in old files


def feature_engineering(df):
    df = df.copy()
    # Hour from hrmn (HHMM, sometimes HH:MM). hrmn is rewritten as homogeneous
    # text to avoid type conflicts when concatenating 20 heterogeneous years.
    if "hrmn" in df.columns:
        hrmn = df["hrmn"].astype(str).str.replace(":", "", regex=False).str.zfill(4)
        df["heure_accident"] = pd.to_numeric(hrmn.str[:2], errors="coerce")
        df["hrmn"] = hrmn
    # User age at the time of the accident
    if {"an", "an_nais"}.issubset(df.columns):
        df["age"] = _full_year(df["an"]) - pd.to_numeric(df["an_nais"], errors="coerce")
        df.loc[(df["age"] < 0) | (df["age"] > 110), "age"] = np.nan
    # Day of week / weekend
    if {"an", "mois", "jour"}.issubset(df.columns):
        dates = pd.to_datetime(
            dict(year=_full_year(df["an"]),
                 month=pd.to_numeric(df["mois"], errors="coerce"),
                 day=pd.to_numeric(df["jour"], errors="coerce")), errors="coerce")
        df["jour_semaine"] = dates.dt.dayofweek       # 0=Monday ... 6=Sunday
        df["weekend"] = df["jour_semaine"].isin([5, 6])
    derived = [c for c in ("heure_accident", "age", "jour_semaine", "weekend") if c in df.columns]
    print("Derived columns:")
    for c in derived:
        n_valid = df[c].notna().sum()
        print(f"  - {c}: {n_valid:,}/{len(df):,} filled ({100 * n_valid / len(df):.1f} %)")
    return df


# ------------------------------------------------------------------
# Coordinates and imputation
# ------------------------------------------------------------------
def to_numeric_comma(series):
    """Numeric conversion handling French decimal commas (e.g. '48,8962100')."""
    if series.dtype == object:
        series = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def clean_coordinates(df):
    """Clean lat/long separately from the imputation step. Two formats coexist:
      - decimal with a French comma ("48,858085");
      - raw integer "degrees x 100000" without comma ("5060813" = 50.60813),
        found in 2010-2018 files, with "0000000" as the "no GPS" sentinel.
    No real coordinate has |value| > 1000, so the raw format is detected by
    magnitude and divided by 100,000. The sentinel 0 becomes NaN explicitly
    (0N 0E is in the Gulf of Guinea, not a real position). Missing coordinates
    stay missing: they are never imputed.
    """
    df = df.copy()
    for col in ("lat", "long"):
        if col not in df.columns:
            continue
        s = to_numeric_comma(df[col])
        rescale = s.abs() > 1000
        if rescale.any():
            s = s.mask(rescale, s / 100_000.0)
            print(f"[clean_coordinates] {col}: {int(rescale.sum()):,} values rescaled "
                  f"(raw x100000 integer format detected).")
        df[col] = s.replace(0.0, np.nan)
    return df


def impute_missing(df):
    df = df.copy()
    if "lat" in df.columns or "long" in df.columns:
        df = clean_coordinates(df)

    cont = [c for c in cfg.CONTINUOUS_OR_ORDINAL_COLS if c in df.columns]
    if cont:
        for c in cont:
            df[c] = to_numeric_comma(df[c])
        # Why these settings: n_jobs=1 because joblib multiprocessing can
        # deadlock silently on Windows/Jupyter; max_samples=5 % per tree keeps
        # single-thread training time independent of the number of rows (a
        # fraction rather than an absolute number because the number of
        # trainable rows varies a lot between columns, e.g. vma is >80 % missing).
        log(f"[impute_missing] Imputing {cont} on {len(df):,} rows "
            f"(n_jobs=1, max_samples=5 % per tree)...")
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=30, max_depth=8, max_samples=0.05,
                random_state=cfg.RANDOM_STATE, n_jobs=1),
            max_iter=3, random_state=cfg.RANDOM_STATE, verbose=2)
        df[cont] = imputer.fit_transform(df[cont])
        log("[impute_missing] Done.")

    cat_cols = [c for c in df.columns
                if c not in cont and not c.startswith("equip_") and df[c].dtype == object]
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(cfg.MISSING_LABEL)
    return df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    help="process only the years FIRST..LAST (default: 2005-2024)")
    ap.add_argument("--no-csv", action="store_true", help="skip the per-year CSV export")
    args = ap.parse_args()
    years = range(args.years[0], args.years[1] + 1) if args.years else cfg.YEARS

    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Steps 1-3: loading, harmonisation, join ===")
    master = build_master_dataset(years)

    print("\n=== Step 4: missing-value normalisation ===")
    master = normalize_missing(master)
    print("Missing rate (top 10, %):")
    print((master.isna().mean() * 100).sort_values(ascending=False).head(10).round(2))

    print("\n=== Step 5: feature engineering ===")
    master = feature_engineering(master)

    print("\n=== Step 6: differentiated imputation ===")
    master = impute_missing(master)

    # Safety net: columns with mixed types across years (int one year, str
    # another) are homogenised to text so the parquet export is reliable.
    for col in master.select_dtypes(include="object").columns:
        master[col] = master[col].astype(str).replace("nan", np.nan)

    # Export per year (the dataset exceeds 4.6 M rows x 66 columns): parquet
    # partitioned by year (readable as one frame or year by year) and one CSV
    # per year.
    # Wipe previous exports: writing into an existing partitioned folder APPENDS new
    # files next to the old ones and silently duplicates every row.
    import shutil
    if cfg.PARQUET_DIR.exists():
        shutil.rmtree(cfg.PARQUET_DIR)
    (cfg.PROCESSED_DIR / "n_rows.txt").write_text(str(len(master)))
    master.to_parquet(cfg.PARQUET_DIR, engine="pyarrow",
                      partition_cols=["annee_dataset"], index=False)
    if not args.no_csv:
        cfg.CSV_DIR.mkdir(parents=True, exist_ok=True)
        print("\nCSV export by year:")
        for year, group in master.groupby("annee_dataset"):
            out = cfg.CSV_DIR / f"baac_master_{int(year)}.csv"
            group.to_csv(out, index=False)
            print(f"  - {out.name}: {len(group):,} rows")

    print(f"\nDone. Final dataset: {master.shape[0]:,} rows x {master.shape[1]} columns.")
    print(f"Parquet: {cfg.PARQUET_DIR}")
    print("Reload with: pd.read_parquet('data/processed/parquet') or filter a year "
          "with filters=[('annee_dataset', '=', 2020)]")


if __name__ == "__main__":
    main()
