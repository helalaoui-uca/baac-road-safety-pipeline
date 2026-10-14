"""
Run the three complementary prediction runs, one after the other, with the right
matplotlib backend (Jupyter's inline backend is invalid outside the notebook).

    python run_extra.py

1. XGBoost-Ordinal on extended_ceinture  -> signed SHAP figures of the ordinal model
2. all models on the "base" feature set
3. all models on the "extended" feature set

Place this file at the root of baac_final (next to run_all.py). A finished run is
not repeated if you pass --skip-done. Log: outputs/run_extra.log
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "04_severity_prediction.py"
LOG = ROOT / "outputs" / "run_extra.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("extended_ceinture", ["--models", "XGBoost-Ordinal"]),
    ("base", []),
    ("extended", []),
]

env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
env["MPLBACKEND"] = "Agg"
env["PYTHONIOENCODING"] = "utf-8"

for fs, extra in RUNS:
    done_file = ROOT / "outputs" / "04_severity_prediction" / fs / "results_comparison.csv"
    if "--skip-done" in sys.argv and fs != "extended_ceinture" and done_file.exists():
        print(f"[skip] {fs} already done")
        continue
    cmd = [sys.executable, str(SCRIPT), "--feature-set", fs, *extra]
    banner = f"\n{'=' * 70}\nRUN: {' '.join(cmd)}\n{'=' * 70}"
    print(banner, flush=True)
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(banner + "\n")
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
        p.wait()
    if p.returncode != 0:
        sys.exit(f"[ERROR] run '{fs}' failed (exit code {p.returncode}). See the lines above.")
    print(f"[ok] {fs} finished in {(time.time() - t0) / 60:.1f} min", flush=True)

print("\nALL DONE. Send me the whole 'outputs' folder (zip), including run_all.log and run_extra.log.")
