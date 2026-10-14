"""Helpers shared by all pipeline steps (logging, loading, severity encoding)."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402


def log(msg):
    """Timestamped, flushed print. If a long run hangs, the last line shows
    exactly which step was running."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_master(columns=None, parquet_dir=None, csv_dir=None):
    """Load the cleaned dataset.

    columns: optional list of columns to load. Loading only what a step needs
    cuts memory use drastically on the ~4.9 M-row dataset. ``annee_dataset`` is
    a hive partition column and is always returned when present.
    """
    parquet_dir = Path(parquet_dir or cfg.PARQUET_DIR)
    csv_dir = Path(csv_dir or cfg.CSV_DIR)

    if parquet_dir.exists():
        log(f"Loading parquet dataset: {parquet_dir}")
        try:
            import pyarrow.dataset as ds
            dataset = ds.dataset(str(parquet_dir), partitioning="hive")
            nf = cfg.PROCESSED_DIR / "n_rows.txt"
            if nf.exists() and dataset.count_rows() != int(nf.read_text()):
                raise RuntimeError(
                    f"Parquet folder holds {dataset.count_rows():,} rows but step 01 wrote "
                    f"{int(nf.read_text()):,} (stale files from an earlier run). "
                    f"Delete {parquet_dir} and re-run step 01.")
            if columns is None:
                df = dataset.to_table().to_pandas()
            else:
                available = set(dataset.schema.names)
                missing = set(columns) - available
                if missing:
                    log(f"[warning] columns absent from the dataset, ignored: {sorted(missing)}")
                df = dataset.to_table(
                    columns=[c for c in columns if c in available]).to_pandas()
        except ImportError:
            df = pd.read_parquet(parquet_dir)
            if columns is not None:
                df = df[[c for c in columns if c in df.columns]]
    elif csv_dir.exists():
        log(f"Parquet not found, loading yearly CSVs: {csv_dir}")
        frames = []
        for f in sorted(csv_dir.glob("*.csv")):
            usecols = None
            if columns is not None:
                in_file = pd.read_csv(f, nrows=0).columns
                usecols = [c for c in columns if c in in_file]
            frames.append(pd.read_csv(f, usecols=usecols, low_memory=False))
        df = pd.concat(frames, ignore_index=True, sort=False)
    else:
        raise FileNotFoundError(
            f"Neither {parquet_dir} nor {csv_dir} exist. Run "
            f"scripts/01_preprocessing.py first or place the cleaned dataset "
            f"under data/processed/ (see data/README.md).")
    log(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns "
        f"({df.memory_usage(deep=True).sum() / 1e9:.2f} GB in memory).")
    return df


def encode_severity(df, add_labels=False):
    """Add the ordinal severity target (1 unharmed ... 4 killed) and, for
    exploratory analyses, the binary 'serious accident' flag
    (hospitalised or killed)."""
    grav = pd.to_numeric(df["grav"], errors="coerce")
    df[cfg.TARGET_COL] = grav.map(cfg.GRAV_TO_ORDINAL).astype("float32")
    if add_labels:
        df["accident_grave"] = df[cfg.TARGET_COL].isin([3, 4])
    return df
