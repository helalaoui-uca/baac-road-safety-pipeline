"""
Step 03 - Spatial and spatio-temporal segmentation of high-risk zones.

Analyses
    1. Spatial clustering: DBSCAN and HDBSCAN (haversine distance, real km)
    2. Spatio-temporal clustering: ST-DBSCAN (own implementation: no sklearn /
       hdbscan class implements it), in two temporal modes
         - "absolute":    time gap in real time (Birant & Kut, 2007)
         - "hour_of_day": circular gap over 24 h, independent of the date
    3. Exposure normalisation: accidents per municipality, and per 10,000
       inhabitants if a population file is provided (config.EXPOSURE_FILE,
       CSV with columns "com" and "population"). Without it only RAW counts
       are produced, with an explicit warning: raw counts mostly reflect
       population/traffic, not intrinsic risk.

The DBSCAN result (outputs/03_spatial_segmentation/dbscan_clusters.parquet)
provides the `cluster_dbscan` variable used by the "geo" prediction variant
(steps 04/05).

Usage:
    python scripts/03_spatial_segmentation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

import config as cfg
from baac.common import load_master, log

MAX_CLUSTERS_SHOWN = 20


# ------------------------------------------------------------------
# Preparation
# ------------------------------------------------------------------
def assign_territory(lat, long):
    for name, (lat_min, lat_max, long_min, long_max) in cfg.TERRITORIES.items():
        if lat_min <= lat <= lat_max and long_min <= long <= long_max:
            return name
    return None


def prepare_spatial_data(df):
    """Keep rows with valid GPS and de-duplicate at accident level (num_acc):
    lat/long/date are identical for every user of an accident, so clustering
    must run on accidents, not users."""
    df = df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["lat", "long"])
    df = df[~((df["lat"] == 0) & (df["long"] == 0))]   # (0, 0) = "unknown" sentinel

    # One bounding box per territory rather than a single national box: a single
    # box lets corrupted coordinates through (e.g. a CSV parsing bug that puts
    # another field - hour, weekday - in lat/long).
    df["territory"] = [assign_territory(la, lo) for la, lo in zip(df["lat"], df["long"])]
    n_out = int(df["territory"].isna().sum())
    if n_out:
        print(f"[!] {n_out:,} rows rejected: outside every known French territory "
              f"(coordinates probably corrupted).")
    df = df.dropna(subset=["territory"])
    if "num_acc" in df.columns:
        df = df.drop_duplicates(subset=["num_acc"])

    print(f"Valid spatial data: {len(df):,} geolocated accidents "
          f"({100 * len(df) / before:.1f} % of starting rows).")
    print("By territory:")
    print(df["territory"].value_counts().to_string())

    if {"an", "mois", "jour"}.issubset(df.columns):
        an = pd.to_numeric(df["an"], errors="coerce")
        an = np.where(an < 100, an + 2000, an)
        hour = pd.to_numeric(df.get("heure_accident"), errors="coerce").fillna(0)
        df["datetime"] = pd.to_datetime(
            dict(year=an, month=pd.to_numeric(df["mois"], errors="coerce"),
                 day=pd.to_numeric(df["jour"], errors="coerce"), hour=hour),
            errors="coerce")
        df = df.dropna(subset=["datetime"])
    return df.reset_index(drop=True)


def to_radians(df):
    return np.radians(df[["lat", "long"]].to_numpy())


def _sample(df, size, name):
    if size is None or len(df) <= size:
        return df
    print(f"[{name}] Sample of {size:,} accidents used (of {len(df):,}).")
    return df.sample(size, random_state=cfg.RANDOM_STATE)


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
def plot_clusters(sub, label_col, title, out_path, max_shown=MAX_CLUSTERS_SHOWN):
    """Two figures: all territories, and mainland France only (a single
    world-scale plot squashes the mainland, which holds most accidents)."""
    def _draw(data, ttl, path):
        plt.figure(figsize=(8, 8))
        noise = data[data[label_col] == -1]
        plt.scatter(noise["long"], noise["lat"], s=2, c="lightgray", alpha=0.3, label="Noise")
        top = data[data[label_col] != -1][label_col].value_counts().head(max_shown).index
        clustered = data[data[label_col].isin(top)]
        plt.scatter(clustered["long"], clustered["lat"], s=3, c=clustered[label_col],
                    cmap="tab20", alpha=0.7)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(ttl)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    _draw(sub, f"{title} (top {max_shown} clusters by size)", out_path)
    if "territory" in sub.columns:
        mainland = sub[sub["territory"] == cfg.MAINLAND]
        if len(mainland):
            _draw(mainland, f"{title} - mainland France (top {max_shown} clusters)",
                  out_path.with_name(out_path.stem + "_mainland" + out_path.suffix))


def _summary(labels, name, extra=""):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[{name}] {n_clusters} clusters, {(labels == -1).mean() * 100:.1f} % noise {extra}")


# ------------------------------------------------------------------
# 1. DBSCAN
# ------------------------------------------------------------------
def dbscan_clustering(df, out):
    """The haversine neighbourhood query (ball_tree) materialises every point's
    neighbour list within eps. In very dense areas (large city centres) a point
    can have thousands of neighbours over 20 years, hence a MemoryError risk
    even with a modest sample: on failure the sample is halved and the run
    retried instead of crashing the whole script."""
    size = cfg.DBSCAN_SAMPLE_SIZE
    eps_rad = cfg.DBSCAN_EPS_KM / cfg.EARTH_RADIUS_KM
    while True:
        sub = _sample(df, size, "DBSCAN")
        model = DBSCAN(eps=eps_rad, min_samples=cfg.DBSCAN_MIN_SAMPLES,
                       metric="haversine", algorithm="ball_tree")
        try:
            labels = model.fit_predict(to_radians(sub))
            break
        except MemoryError:
            size = (size or len(df)) // 2
            if size < cfg.DBSCAN_MIN_SAMPLE_SIZE_RETRY:
                print("[DBSCAN] Persistent MemoryError: reduce the sample manually or "
                      "process very dense areas separately.")
                raise
            print(f"[DBSCAN] MemoryError - retrying with {size:,} points.")
    sub = sub.copy()
    sub["cluster_dbscan"] = labels
    _summary(labels, "DBSCAN",
             f"(eps={cfg.DBSCAN_EPS_KM} km, min_samples={cfg.DBSCAN_MIN_SAMPLES}, "
             f"sample={len(sub):,} accidents)")
    sub.to_parquet(out / "dbscan_clusters.parquet", index=False)
    plot_clusters(sub, "cluster_dbscan", "Spatial clustering - DBSCAN",
                  out / "dbscan_clusters.png")
    return sub


# ------------------------------------------------------------------
# 2. HDBSCAN
# ------------------------------------------------------------------
def hdbscan_clustering(df, out):
    try:
        import hdbscan
    except ImportError as e:
        print(f"[!] hdbscan not installed ({e}) - step skipped.")
        return None
    sub = _sample(df, cfg.HDBSCAN_SAMPLE_SIZE, "HDBSCAN").copy()
    model = hdbscan.HDBSCAN(min_cluster_size=cfg.HDBSCAN_MIN_CLUSTER_SIZE, metric="haversine")
    sub["cluster_hdbscan"] = model.fit_predict(to_radians(sub))
    _summary(sub["cluster_hdbscan"].to_numpy(), "HDBSCAN",
             f"(min_cluster_size={cfg.HDBSCAN_MIN_CLUSTER_SIZE})")
    sub.to_parquet(out / "hdbscan_clusters.parquet", index=False)
    plot_clusters(sub, "cluster_hdbscan", "Spatial clustering - HDBSCAN",
                  out / "hdbscan_clusters.png")
    return sub


# ------------------------------------------------------------------
# 3. ST-DBSCAN (own implementation)
# ------------------------------------------------------------------
def st_dbscan(df, out, temporal_mode="absolute"):
    """ST-DBSCAN (Birant & Kut, 2007): two points are neighbours when their
    spatial distance AND their temporal gap are each under a dedicated
    threshold. Implemented as a region-expansion DBSCAN whose spatial query
    (BallTree / haversine) is filtered afterwards by the temporal gap.

    temporal_mode="absolute": gap in real time (same-day clusters: pile-ups,
        common weather episodes). Very restrictive over 20 years, so few
        clusters are expected - that is not an error.
    temporal_mode="hour_of_day": circular gap over 24 h, regardless of the
        date; finds recurring time-of-day hotspots (e.g. a junction that is
        dangerous at rush hour).
    """
    if temporal_mode not in ("absolute", "hour_of_day"):
        raise ValueError('temporal_mode must be "absolute" or "hour_of_day".')
    sub = _sample(df, cfg.ST_DBSCAN_SAMPLE_SIZE, "ST-DBSCAN")
    sub = sub.sort_values("datetime").reset_index(drop=True)
    coords = to_radians(sub)
    eps_rad = cfg.ST_EPS_SPATIAL_KM / cfg.EARTH_RADIUS_KM

    if temporal_mode == "absolute":
        times = sub["datetime"].astype("int64").to_numpy() / 3.6e12   # hours since epoch

        def gap(cand, i):
            return np.abs(times[cand] - times[i])
    else:
        hours = (sub["datetime"].dt.hour + sub["datetime"].dt.minute / 60.0).to_numpy()

        def gap(cand, i):
            raw = np.abs(hours[cand] - hours[i])
            return np.minimum(raw, 24.0 - raw)     # 23h and 1h are 2 h apart, not 22

    spatial_neighbors = BallTree(coords, metric="haversine").query_radius(coords, r=eps_rad)
    n = len(sub)
    labels = np.full(n, -2, dtype=int)   # -2 unvisited, -1 noise, >=0 cluster id
    cluster_id = -1

    def st_neighbors(i):
        cand = spatial_neighbors[i]
        return cand[gap(cand, i) <= cfg.ST_EPS_TEMPORAL_HOURS]

    for i in range(n):
        if labels[i] != -2:
            continue
        neighbors = st_neighbors(i)
        if len(neighbors) < cfg.ST_MIN_SAMPLES:
            labels[i] = -1
            continue
        cluster_id += 1
        labels[i] = cluster_id
        seeds, pos = list(neighbors), 0
        while pos < len(seeds):
            j = seeds[pos]
            pos += 1
            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] != -2:
                continue
            labels[j] = cluster_id
            j_neighbors = st_neighbors(j)
            if len(j_neighbors) >= cfg.ST_MIN_SAMPLES:
                seeds.extend(int(k) for k in j_neighbors if labels[k] == -2)

    sub = sub.copy()
    sub["cluster_stdbscan"] = labels
    _summary(labels, f"ST-DBSCAN, mode={temporal_mode}",
             f"(eps_spatial={cfg.ST_EPS_SPATIAL_KM} km, "
             f"eps_temporal={cfg.ST_EPS_TEMPORAL_HOURS} h, min_samples={cfg.ST_MIN_SAMPLES})")
    suffix = "" if temporal_mode == "absolute" else "_hour_of_day"
    sub.to_parquet(out / f"stdbscan_clusters{suffix}.parquet", index=False)
    title_suffix = "" if temporal_mode == "absolute" else " (hour of day)"
    plot_clusters(sub, "cluster_stdbscan",
                  f"Spatio-temporal clustering - ST-DBSCAN{title_suffix}",
                  out / f"stdbscan_clusters{suffix}.png")
    return sub


# ------------------------------------------------------------------
# 4. Exposure normalisation
# ------------------------------------------------------------------
def build_insee_code(df):
    """Rebuild the full INSEE municipality code from `dep` and `com`.

    In BAAC, `com` is a 3-digit code that is only unique WITHIN a department,
    not a national identifier: grouping on `com` alone merges unrelated
    municipalities of different departments (observed: com=55 gathered 13,873
    accidents). Special cases:
      - Paris, Lyon, Marseille: `com` already holds the full 5-digit
        arrondissement code (75101-75120, 69381-69389, 13201-13216): kept as is.
      - Corsica: `com` sometimes already holds the full lettered code ("2A004",
        "2B033"); other years code the department as "2A0"/"2B0" or "201"/"202":
        all normalised to "2A"/"2B".
      - Mainland (otherwise): `dep` normalised to 2 digits (old years append a
        trailing zero, e.g. "590" for Nord) + `com` zero-padded to 3 digits.
      - DROM (971-976): municipality coded on 2 digits (not 3) in INSEE
        nomenclature -> `dep` (3 digits) + `com` zero-padded to 2 digits.
      - Columns cast to float elsewhere ("202346.0") have the ".0" stripped.
    """
    dep_raw = df["dep"].astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    com_raw = df["com"].astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    is_complete = com_raw.str.fullmatch(r"\d{5}") | com_raw.str.fullmatch(r"2[AB]\d{3}")
    is_drom = dep_raw.str[:3].isin(["971", "972", "973", "974", "975", "976"])

    def norm_dep(d):
        if d in ("2A0", "2B0", "201", "202"):
            return "2A" if d in ("2A0", "201") else "2B"
        if len(d) == 3 and d.isdigit() and d.endswith("0") and 10 <= int(d[:2]) <= 95:
            return d[:2]
        return d

    dep_norm = dep_raw.map(norm_dep)
    com_padded = np.where(is_drom, com_raw.str.zfill(2), com_raw.str.zfill(3))
    insee = np.where(is_complete, com_raw.str.zfill(5), dep_norm + com_padded)
    return pd.Series(insee, index=df.index, name="insee_code")


def exposure_normalization(df, out):
    if "com" not in df.columns:
        print("[Exposure] Column 'com' missing - normalisation impossible.")
        return None
    df = df.copy()
    if "dep" in df.columns:
        df["insee_code"] = build_insee_code(df)
    else:
        print("[Exposure] Column 'dep' missing: grouping on 'com' alone (risk of "
              "merging municipalities from different departments).")
        df["insee_code"] = df["com"].astype(str).str.strip()

    counts = (df.groupby("insee_code").size().rename("n_accidents").reset_index()
              .rename(columns={"insee_code": "com"})
              .sort_values("n_accidents", ascending=False))

    exposure_file = cfg.EXPOSURE_FILE
    if exposure_file is None or not Path(exposure_file).exists():
        print("\n[!] No exposure file (config.EXPOSURE_FILE). The ranking below is RAW "
              "accident density per municipality: it mostly reflects population and "
              "traffic, NOT a higher intrinsic risk. Do not read it as 'high-risk "
              "zones' without normalising by an exposure indicator.")
        counts.to_csv(out / "accidents_per_municipality_raw.csv", index=False)
        print("\nTop 10 municipalities by raw accident count:")
        print(counts.head(10).to_string(index=False))
        return counts

    exposure = pd.read_csv(exposure_file, dtype={"com": str})
    if not {"com", "population"}.issubset(exposure.columns):
        raise ValueError("The exposure file must contain columns 'com' and 'population'.")
    counts["com"] = counts["com"].astype(str).str.strip()
    exposure["com"] = exposure["com"].astype(str).str.strip()
    merged = counts.merge(exposure, on="com", how="left")
    merged["rate_per_10k_inhabitants"] = merged["n_accidents"] / merged["population"] * 10_000
    merged = merged.sort_values("rate_per_10k_inhabitants", ascending=False)
    merged.to_csv(out / "accidents_per_municipality_normalised.csv", index=False)
    print("\nTop 10 municipalities by accident rate per 10,000 inhabitants:")
    print(merged.head(10).to_string(index=False))
    print("\nTop 10 by raw count (for comparison):")
    print(counts.head(10).to_string(index=False))
    return merged


def main():
    out = cfg.SPATIAL_DIR
    out.mkdir(parents=True, exist_ok=True)

    cols = ["lat", "long", "num_acc", "annee_dataset", "an", "mois", "jour",
            "heure_accident", "dep", "com"]
    spatial_df = prepare_spatial_data(load_master(columns=cols))

    log("=== 1. Spatial clustering - DBSCAN ===")
    dbscan_clustering(spatial_df, out)
    log("=== 2. Spatial clustering - HDBSCAN ===")
    hdbscan_clustering(spatial_df, out)
    log("=== 3. Spatio-temporal clustering - ST-DBSCAN ===")
    if "datetime" in spatial_df.columns:
        st_dbscan(spatial_df, out, temporal_mode="absolute")
        st_dbscan(spatial_df, out, temporal_mode="hour_of_day")
    else:
        print("[!] Columns an/mois/jour missing - ST-DBSCAN skipped.")
    log("=== 4. Exposure normalisation ===")
    exposure_normalization(spatial_df, out)
    log(f"Done. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
