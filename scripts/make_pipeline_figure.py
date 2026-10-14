"""
Draw the experimental-pipeline summary figure (figures/pipeline_overview.png and
a grayscale version, figures/pipeline_overview_gray.png).

Usage:
    python scripts/make_pipeline_figure.py
"""

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import config as cfg

TITLE = "Experimental pipeline for severity prediction"
STEPS = ["1  Target & data variants", "2  Temporal validation", "3  Feature-set selection",
         "4  Models & class imbalance", "5  Evaluation & robustness", "6  Interpretability"]
BLOCKS = [
    ["BAAC 2005-2024 (2.89 M rows)\nTarget: ordinal severity\n1 unharmed · 2 light · 3 hospitalised · 4 killed",
     "Variant A \"full\"\nall rows, no spatial variable",
     "Variant B \"geo\"\nusers of the DBSCAN-sampled accidents + DBSCAN cluster"],
    ["Train 2005-2021\nTest 2022-2024",
     "Why: non-stationarity and\nnomenclature breaks\nno random split"],
    ["base: 21 categorical factors\n+ 7 continuous (28/29 features)",
     "extended: + 5 safety equipments\n(coded since 2019 only) -> rejected",
     "extended_belt: + 9 variables\n(seatbelt, user, seat, ...) -> retained"],
    ["Class weights: RandomForest, LightGBM, XGBoost",
     "XGBoost-Ordinal: Frank & Hall,\nsqrt-damped rebalancing\n(full ratio degraded macro-F1)",
     "XGBoost-SMOTE: SMOTENC on\nstratified 300 k subsample"],
    ["Macro-F1 (main metric)\nweighted-F1, accuracy, per-class",
     "Ordinal MAE · weighted kappa\nrecall/F1 of class \"killed\"",
     "5 seeds (mean ± sd)\nbootstrap CI + paired differences"],
    ["SHAP on 5,000 test samples\n(best model per variant)"],
]
NOTE = "Explanatory models: circumstances known after the accident"

PALETTES = {
    "color": (["#E8EEF7", "#E8F4EC", "#FFF4E0", "#F6E8F4", "#E8F4F4", "#F2F2F2"],
              ["#3C6AA8", "#3A8A5A", "#C98A1B", "#8A3A82", "#2F8A8A", "#666666"]),
    "gray": (["#F4F4F4", "#EAEAEA", "#E0E0E0", "#D6D6D6", "#CCCCCC", "#C2C2C2"],
             ["#2A2A2A", "#3F3F3F", "#545454", "#686868", "#7C7C7C", "#909090"]),
}


def wrap(text, width):
    return "\n".join("\n".join(textwrap.wrap(l, width)) if l.strip() else l
                     for l in text.split("\n"))


def draw(palette, path):
    fills, edges = PALETTES[palette]
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8)
    ax.axis("off")
    w, gap = 2.95, 0.3
    xs = [0.2 + i * (w + gap) for i in range(6)]
    top, bh, vgap = 7.0, 1.55, 0.2
    ax.text(10, 7.65, TITLE, ha="center", va="center", fontsize=17, fontweight="bold")
    for i, (x, head, blocks) in enumerate(zip(xs, STEPS, BLOCKS)):
        ax.add_patch(FancyBboxPatch((x, top - 0.75), w, 0.8,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=edges[i], ec=edges[i]))
        ax.text(x + w / 2, top - 0.35, wrap(head, 24), ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
        y = top - 1.0
        for t in blocks:
            ax.add_patch(FancyBboxPatch((x, y - bh), w, bh,
                                        boxstyle="round,pad=0.02,rounding_size=0.1",
                                        fc=fills[i], ec=edges[i], lw=1.3))
            ax.text(x + w / 2, y - bh / 2, wrap(t, 34), ha="center", va="center", fontsize=9.5)
            y -= bh + vgap
        if i < 5:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, top - 0.35), (xs[i + 1] - 0.03, top - 0.35),
                                         arrowstyle="-|>", mutation_scale=16, color="#444", lw=1.6))
    ax.text(10, 0.45, NOTE, ha="center", va="center", fontsize=11, style="italic", color="#444")
    plt.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    out = cfg.ROOT / "figures"
    out.mkdir(exist_ok=True)
    draw("color", out / "pipeline_overview.png")
    draw("gray", out / "pipeline_overview_gray.png")
    print(f"Saved in {out}")
