"""
generate_permutation_importance.py
-----------------------------------
Computes permutation-based feature importance for the XGBoost model
over all 406 dimensions (384 MiniLM-L6-v2 + 22 static features).

Permutation importance is model-agnostic and avoids the impurity bias
of gain-based importance. It measures how much the macro-F1 score drops
when each feature is randomly shuffled. The model is trained exactly as in the
benchmark (Gaussian augmentation + class weights, XGBOOST_PARAMS) on each training
split of the seed-42 Stratified Grouped 5-fold CV, and importance is measured on the
corresponding held-out fold (10 shuffles per feature); the reported value is the
mean over the five held-out folds, the error bar the SD across folds.

Outputs (saved to results/):
  - permutation_importance.json          all 406 features ranked
  - permutation_importance_top20.json    top-20 summary for paper
  - plots/perm_importance_top20.png      horizontal bar chart (this is fig_perm_importance.png in the paper)

Run: python generate_permutation_importance.py
"""

import io
import json
import sys
from contextlib import contextmanager
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from train.data_loader import load_records
from train.evaluate import get_cv_splits, get_sample_weights
from train.features.fused_features import fused_features, apply_gaussian_augmentation, STATIC_FEATURE_NAMES
from train.config import RESULTS_DIR, RANDOM_SEED, NUM_LABELS, XGBOOST_PARAMS


FEATURE_NAMES = [f"MiniLM_{i}" for i in range(384)] + list(STATIC_FEATURE_NAMES)
assert len(FEATURE_NAMES) == 406


# ---------------------------------------------------------------------------
# Tee helper
# ---------------------------------------------------------------------------
class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()

@contextmanager
def capture_stdout(buf):
    old = sys.stdout
    sys.stdout = _Tee(old, buf)
    try:
        yield
    finally:
        sys.stdout = old


def macro_f1_scorer(estimator, X, y):
    """Custom scorer: macro-F1 (used by permutation_importance)."""
    y_pred = estimator.predict(X)
    return f1_score(y, y_pred, average="macro", zero_division=0)


def main():
    print("=" * 66)
    print("  Permutation Feature Importance -- XGBoost (406-dim)")
    print("=" * 66)

    # -- Load data & extract features -----------------------------------------
    print("\n[1/4] Loading data ...")
    texts, labels, group_ids, source_datasets, filenames = load_records()
    labels_arr = np.array(labels)

    print("[2/4] Extracting 406-dimensional features (MiniLM + Static) ...")
    X_all = fused_features(texts, device='cpu')

    # Importance is measured on held-out data of the SAME grouped folds the models are
    # evaluated on (seed RANDOM_SEED), so no clone family is on both sides of a split.
    splits = get_cv_splits(labels, group_ids, seed=RANDOM_SEED)

    fold_imps, baselines = [], []
    for k, (tr, te) in enumerate(splits, 1):
        # -- Train XGBoost exactly as in the benchmark (augmentation + class weights)
        print(f"[3/4] Fold {k}/{len(splits)}: training XGBoost on {len(tr)} contracts ...")
        # same class- and source-balanced weights as the benchmark; noisy copies inherit them
        sw = get_sample_weights(labels_arr[tr], np.asarray(source_datasets, dtype=object)[tr])
        X_aug, y_aug, w_aug = apply_gaussian_augmentation(X_all[tr], labels_arr[tr], NUM_LABELS,
                                                          seed=RANDOM_SEED, weights=sw)
        clf = xgb.XGBClassifier(**{k_: v for k_, v in XGBOOST_PARAMS.items() if k_ != "eval_metric"})
        clf.fit(X_aug, y_aug, sample_weight=w_aug)

        baselines.append(macro_f1_scorer(clf, X_all[te], labels_arr[te]))
        print(f"   Baseline macro-F1 on held-out fold: {baselines[-1]:.4f}")

        # -- Permutation importance on the held-out fold -------------------------
        print(f"[4/4] Fold {k}: permutation importance (n_repeats=10) ...")
        result = permutation_importance(
            clf, X_all[te], labels_arr[te],
            scoring=macro_f1_scorer,
            n_repeats=10,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        fold_imps.append(result.importances_mean)

    baseline_f1 = float(np.mean(baselines))
    perm_mean  = np.mean(fold_imps, axis=0)   # shape: (406,) mean over the 5 held-out folds
    perm_std   = np.std(fold_imps, axis=0)    # SD across folds

    # -- Sort & build output dict ----------------------------------------------
    sorted_idx = np.argsort(perm_mean)[::-1]

    importance_dict = {
        FEATURE_NAMES[i]: {
            "mean_decrease_f1": round(float(perm_mean[i]), 6),
            "std":              round(float(perm_std[i]),  6),
            "rank":             int(np.where(sorted_idx == i)[0][0]) + 1,
            "feature_type":     "semantic_embedding" if i < 384 else "static_feature",
        }
        for i in range(406)
    }

    top20_idx   = sorted_idx[:20]
    top20_names = [FEATURE_NAMES[i] for i in top20_idx]
    top20_means = perm_mean[top20_idx]
    top20_stds  = perm_std[top20_idx]

    top20_dict = {
        FEATURE_NAMES[i]: {
            "rank":             int(rank + 1),
            "mean_decrease_f1": round(float(perm_mean[i]), 6),
            "std":              round(float(perm_std[i]),  6),
            "feature_type":     "semantic_embedding" if i < 384 else "static_feature",
        }
        for rank, i in enumerate(top20_idx)
    }

    # -- Save JSON -------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_full = RESULTS_DIR / "permutation_importance.json"
    with out_full.open("w", encoding="utf-8") as f:
        json.dump(importance_dict, f, indent=2)
    print(f"\n  [OK]  Full importance saved -> {out_full}")

    out_top20 = RESULTS_DIR / "permutation_importance_top20.json"
    with out_top20.open("w", encoding="utf-8") as f:
        json.dump(top20_dict, f, indent=2)
    print(f"  [OK]  Top-20 saved -> {out_top20}")

    # -- Plot 1: Top-20 bar chart ----------------------------------------------
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ['#4e79a7' if n.startswith("MiniLM") else '#f28e2b'
              for n in reversed(top20_names)]
    y_pos  = range(20)
    ax.barh(y_pos, top20_means[::-1], xerr=top20_stds[::-1],
            color=colors, edgecolor='white', linewidth=0.4,
            error_kw=dict(ecolor='#555', capsize=3, lw=1))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(reversed(top20_names)), fontsize=9)
    ax.set_xlabel("Mean drop in macro-F1 when the feature is shuffled "
                  "(10 shuffles per held-out fold; mean over 5 grouped folds, error bar = SD across folds)",
                  fontsize=9)
    ax.set_title(
        "Permutation importance, top 20 features\n"
        "XGBoost on the 406-d fused vector (384 MiniLM + 22 static), "
        "held-out grouped CV folds",
        fontsize=10, pad=12
    )
    legend_handles = [
        mpatches.Patch(color='#4e79a7', label='Semantic embedding (MiniLM, 384-d)'),
        mpatches.Patch(color='#f28e2b', label='Static / regex feature (22-d)'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    p1 = plots_dir / "perm_importance_top20.png"
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK]  Plot saved -> {p1}")

    # -- Print & save top-10 summary -------------------------------------------
    buf = io.StringIO()
    with capture_stdout(buf):
        print("\n-- Top-10 Permutation-Important Features ----------------------------")
        print(f"  {'Rank':<6} {'Feature':<40} {'Mean dF1':>10}  {'Std':>8}  Type")
        print("  " + "-" * 72)
        for rank, i in enumerate(top20_idx[:10], 1):
            kind = "semantic" if i < 384 else "static"
            print(f"  {rank:<6} {FEATURE_NAMES[i]:<40} {perm_mean[i]:>10.4f}  "
                  f"{perm_std[i]:>8.4f}  {kind}")

        print(f"\n  Baseline macro-F1 (mean over held-out folds): {baseline_f1:.4f}")
        print("  Interpretation: features with mean decrease > 0 are predictive;")
        print("  negative values indicate the feature adds noise.")

    report_out = RESULTS_DIR / "permutation_importance_report.txt"
    with report_out.open("w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"  [OK]  Report saved -> {report_out}")


if __name__ == "__main__":
    main()
