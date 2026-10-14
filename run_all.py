"""
Run the WHOLE BAAC pipeline in one command, with automatic resume.

    python run_all.py --raw-dir "C:\\path\\to\\raw_csv_folder"

Steps (each is skipped if its output already exists, unless --force):
    01 preprocessing        -> data/processed/parquet
    00 data overview        -> outputs/00_data_overview
    02 risk factors         -> outputs/02_risk_factors      (--skip-rules to go faster)
    03 spatial segmentation -> outputs/03_spatial_segmentation
    04 severity prediction  -> outputs/04_severity_prediction/<feature_set>
    05 robustness           -> .../<feature_set>/robustness

Raw data: either <raw-dir>/<year>/*.csv or a flat folder with files such as
usagers_2016.csv, vehicules_2016.csv, caracteristiques_2016.csv, lieux_2016.csv.

Useful options:
    --steps 1 4 5        run only some steps
    --force              recompute even if outputs exist
    --with-base          also run the "base" feature set (to measure the gain)
    --years 2005 2024    restrict the years of step 01
    --skip-rules         skip association rules (step 02, slow)
    --quick              fast smoke test: 2 seeds, 1 model per step 05

A full log is written to outputs/run_all.log.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import config as cfg

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
LOG = cfg.OUTPUT_ROOT / "run_all.log"


def banner(msg):
    line = f"\n{'=' * 70}\n{msg}\n{'=' * 70}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(script, args=()):
    cmd = [sys.executable, str(SCRIPTS / script), *map(str, args)]
    print("$", " ".join(cmd), flush=True)
    t0 = time.time()
    # Jupyter sets MPLBACKEND to its inline backend, invalid outside a notebook.
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    with open(LOG, "a", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", env=env)
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
        proc.wait()
    dt = (time.time() - t0) / 60
    if proc.returncode != 0:
        sys.exit(f"\n[ERROR] {script} failed (exit code {proc.returncode}). "
                 f"Fix the problem and re-run the same command: finished steps are skipped.")
    print(f"[ok] {script} finished in {dt:.1f} min", flush=True)


def done(path):
    path = Path(path)
    return path.exists() and (path.is_file() or any(path.iterdir()))


def check_rows():
    """Sanity check: rows per year of the cleaned dataset (no duplication)."""
    import pandas as pd
    import pyarrow.dataset as ds
    if not Path(cfg.PARQUET_DIR).exists():
        sys.exit(f"[ERROR] {cfg.PARQUET_DIR} does not exist: step 01 did not write the "
                 f"cleaned dataset. Re-run with --steps 1 --force.")
    dataset = ds.dataset(str(cfg.PARQUET_DIR), partitioning="hive")
    df = dataset.to_table(columns=["annee_dataset"]).to_pandas()
    counts = df["annee_dataset"].astype(int).value_counts().sort_index()
    banner(f"Rows per year (total {len(df):,})")
    nf = cfg.PROCESSED_DIR / "n_rows.txt"
    if nf.exists() and int(nf.read_text()) != len(df):
        sys.exit(f"[ERROR] the parquet folder holds {len(df):,} rows but step 01 wrote "
                 f"{int(nf.read_text()):,}: stale files from an earlier run. Delete "
                 f"{cfg.PARQUET_DIR} and re-run with --force.")
    print(counts.to_string(), flush=True)
    if cfg.JOIN_ON_NUM_VEH and counts.max() > 1.6 * counts.median():
        print("[WARNING] one year has far more rows than the others: "
              "check for join duplication.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--raw-dir", help="folder holding the raw BAAC CSV files")
    ap.add_argument("--steps", nargs="+", type=int, default=[1, 0, 2, 3, 4, 5],
                    choices=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--with-base", action="store_true")
    ap.add_argument("--years", nargs=2, type=int)
    ap.add_argument("--skip-rules", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    if a.raw_dir:                       # config reads BAAC_RAW_DIR at import time
        import os
        os.environ["BAAC_RAW_DIR"] = a.raw_dir
        cfg.RAW_DIR = Path(a.raw_dir)
    cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    banner(f"BAAC pipeline | raw: {cfg.RAW_DIR} | JOIN_ON_NUM_VEH={cfg.JOIN_ON_NUM_VEH}")
    if not cfg.JOIN_ON_NUM_VEH:
        print("[WARNING] JOIN_ON_NUM_VEH is False: rows are duplicated before 2019.")

    fsets = [cfg.DEFAULT_FEATURE_SET] + (["base"] if a.with_base else [])
    steps = a.steps

    if 1 in steps:
        banner("STEP 01 - preprocessing")
        if done(cfg.PARQUET_DIR) and not a.force:
            print("[skip] cleaned dataset already exists (use --force to rebuild).")
        else:
            run("01_preprocessing.py", ["--years", *a.years] if a.years else [])
        check_rows()

    if 0 in steps:
        banner("STEP 00 - data overview")
        if done(cfg.OVERVIEW_DIR) and not a.force:
            print("[skip] already done.")
        else:
            run("00_data_overview.py")

    if 2 in steps:
        banner("STEP 02 - risk factors")
        if done(cfg.RISK_FACTORS_DIR) and not a.force:
            print("[skip] already done.")
        else:
            run("02_risk_factors.py", ["--skip-rules"] if a.skip_rules else [])

    if 3 in steps:
        banner("STEP 03 - spatial segmentation")
        if done(cfg.SPATIAL_CLUSTERS_PATH) and not a.force:
            print("[skip] already done.")
        else:
            run("03_spatial_segmentation.py")

    for fs in fsets:
        out = cfg.PREDICTION_DIR / fs
        if 4 in steps:
            banner(f"STEP 04 - severity prediction ({fs})")
            if done(out / "results_comparison.csv") and not a.force:
                print("[skip] already done.")
            else:
                run("04_severity_prediction.py", ["--feature-set", fs])
        if 5 in steps and fs == cfg.DEFAULT_FEATURE_SET:
            banner(f"STEP 05 - robustness ({fs})")
            if done(out / "robustness" / "robustness_bootstrap_diff.csv") and not a.force:
                print("[skip] already done.")
            else:
                extra = ["--seeds", "0", "1", "--models", "XGBoost-Ordinal", "XGBoost"] \
                    if a.quick else []
                run("05_robustness.py", ["--feature-set", fs, *extra])

    banner("ALL DONE. Results in: " + str(cfg.OUTPUT_ROOT.resolve()))


if __name__ == "__main__":
    main()
