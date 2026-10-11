"""
plot_paper_figures.py
---------------------
Regenerates the paper's result figures from results/analysis/analysis.json
(written by analyze_results.py) and the permutation-importance plot:

    results/paper_figures/fig_model_comparison.png   all-contract / SolidiFI / real-world macro-F1
    results/paper_figures/fig_cm_smartbert_combined.png  SmartBERT confusion matrix
    results/paper_figures/fig_perm_importance.png    copy of results/plots/perm_importance_top20.png

Run after analyze_results.py and generate_permutation_importance.py.
"""
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train.config import LABELS, RESULTS_DIR

OUT = RESULTS_DIR / "paper_figures"
NAMES = {
    "xgboost": "XGBoost", "smartembed": "SmartEmbed", "eth2vec": "ETH2Vec", "deepsolid": "DeepSolid",
    "hybridvuldetect": "HybridVulDetect", "codebert": "CodeBERT", "smartbert": "SmartBERT", "gnn": "Contract-GCN",
}
CLASS_NAMES = ["Access\nControl", "Arithmetic", "Front-\nRunning", "Reentrancy", "Time\nManip.", "Unchecked\nLL Calls"]
SERIES = [("all", "All contracts", "#2a78d6"), ("solidifi", "SolidiFI", "#eb6834"), ("realworld", "Real-world", "#1baf7a")]
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"


def model_comparison(A):
    sl = A["source_level"]["by_model"]
    models = sorted(A["models"], key=lambda m: -sl[m]["all"]["mean"])
    x = np.arange(len(models))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for k, (key, label, color) in enumerate(SERIES):
        means = np.array([sl[m][key]["mean"] for m in models])
        lo = means - np.array([sl[m][key]["ci95_bootstrap"][0] for m in models])
        hi = np.array([sl[m][key]["ci95_bootstrap"][1] for m in models]) - means
        ax.bar(x + (k - 1) * (w + 0.02), means, w, color=color, label=label, zorder=2,
               edgecolor="white", linewidth=0.8)
        ax.errorbar(x + (k - 1) * (w + 0.02), means, yerr=[lo, hi], fmt="none",
                    ecolor=MUTED, elinewidth=0.9, capsize=2.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[m] for m in models], fontsize=9, color=INK)
    ax.set_ylabel("Macro-F1", fontsize=10, color=INK)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.08))
    fig.tight_layout()
    fig.savefig(OUT / "fig_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def confusion_smartbert(A):
    cm = np.asarray(A["confusion_matrices_summed_over_repeats"]["smartbert"]["all"], dtype=float) / A["n_repeats"]
    rows = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.imshow(rows, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"]), vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if rows[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=8.5, color=INK)
    one_line = ["Access Control", "Arithmetic", "Front-Running", "Reentrancy", "Time Manip.", "Unchecked LL Calls"]
    ax.set_yticklabels([f"{c} ({int(round(t))})" for c, t in zip(one_line, cm.sum(1))], fontsize=8.5, color=INK)
    ax.set_xlabel("Predicted class", fontsize=10, color=INK)
    ax.set_ylabel("True class (contracts)", fontsize=10, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cm_smartbert_combined.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    global OUT
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Regenerate the paper's result figures.")
    ap.add_argument("--results", default=str(RESULTS_DIR),
                    help="folder holding analysis/analysis.json (e.g. paper_results for the archived run)")
    root = Path(ap.parse_args().results)
    OUT = root / "paper_figures"
    OUT.mkdir(parents=True, exist_ok=True)
    A = json.loads((root / "analysis" / "analysis.json").read_text(encoding="utf-8"))
    model_comparison(A)
    confusion_smartbert(A)
    src = root / "plots" / "perm_importance_top20.png"
    if src.exists():
        shutil.copy2(src, OUT / "fig_perm_importance.png")
    print(f"[OK] figures -> {OUT}")


if __name__ == "__main__":
    main()
