"""train/evaluate.py — Shared CV splits, metrics, and result persistence."""
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Callable

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, accuracy_score, matthews_corrcoef,
)
from sklearn.utils.class_weight import compute_class_weight

from train.config import (
    LABELS, NUM_LABELS, N_SPLITS, get_seed, get_results_dir,
)


# ── Splits ────────────────────────────────────────────────────────────────────

def get_cv_splits(
    labels: List[int],
    group_ids: List[str],
    seed: int | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns list of (train_idx, test_idx) arrays — the same splits are reused
    for every model of a run so the evaluation is paired. ``seed`` defaults to
    the current run seed (train.config.get_seed()). No clone-family group ever
    appears on both sides of a split (asserted).
    """
    seed = get_seed() if seed is None else seed
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    X_dummy = np.zeros(len(labels))
    splits = list(sgkf.split(X_dummy, labels, group_ids))
    g = np.asarray(group_ids, dtype=object)
    for tr, te in splits:
        assert not (set(g[tr]) & set(g[te])), "clone-family group crosses a fold boundary"
    return splits


# ── Class weights ─────────────────────────────────────────────────────────────

def get_class_weights(y_train: List[int]) -> np.ndarray:
    classes = np.arange(NUM_LABELS)
    present = np.unique(y_train)
    if len(present) == NUM_LABELS:
        # All classes present — normal path
        return compute_class_weight("balanced", classes=classes, y=y_train)
    else:
        # Some classes missing from the training split (possible in leave-one-source-out runs).
        # Compute weights only for present classes, assign 1.0 to absent ones.
        weights = np.ones(NUM_LABELS, dtype=float)
        w = compute_class_weight("balanced", classes=present, y=y_train)
        for cls, wt in zip(present, w):
            weights[cls] = wt
        return weights


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    source_datasets: np.ndarray | None = None,
    prefix: str = "",
) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    result: dict = {
        f"{prefix}macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}mcc":      float(matthews_corrcoef(y_true, y_pred)),
    }

    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, label in enumerate(LABELS):
        result[f"{prefix}f1_{label}"] = float(
            per_class[i] if i < len(per_class) else 0.0
        )

    if source_datasets is not None:
        src_arr = np.array(source_datasets)
        for src in ("solidifi", "realworld"):
            mask = src_arr == src
            if mask.sum() > 0:
                result[f"{prefix}{src}_macro_f1"] = float(
                    f1_score(
                        y_true[mask], y_pred[mask],
                        average="macro", zero_division=0,
                    )
                )

    result[f"{prefix}y_true"] = y_true.tolist()
    result[f"{prefix}y_pred"] = y_pred.tolist()
    if source_datasets is not None:
        result[f"{prefix}sources"] = src_arr.tolist()

    return result


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_folds(fold_results: List[dict]) -> dict:
    agg: dict = {}
    skip = {"fold", "n_train", "n_test", "y_true", "y_pred", "sources",
            "test_idx", "test_eligible"}
    numeric_keys = [
        k for k, v in fold_results[0].items()
        if isinstance(v, (int, float)) and k not in skip
    ]
    for key in numeric_keys:
        vals = [f[key] for f in fold_results]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"]  = float(np.std(vals))
    return agg


# ── Persistence ───────────────────────────────────────────────────────────────

def save_results(model_name: str, fold_results: List[dict], agg: dict) -> Path:
    out_dir = get_results_dir() / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fold_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"model": model_name, "seed": get_seed(), "folds": fold_results, "aggregate": agg},
            f, indent=2,
        )
    return out_path


# ── High-level CV runner for sklearn-style models ─────────────────────────────

def run_sklearn_cv(
    model_name: str,
    build_features_fn: Callable,      # build_features_fn(texts) → X (array/sparse)
    train_model_fn: Callable,         # train_model_fn(X_tr, y_tr, cw) → clf
    predict_fn: Callable,             # predict_fn(clf, X_te) → List[int]
    texts: List[str],
    labels: List[int],
    source_datasets: List[str],
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> dict:
    """
    Generic CV loop for XGBoost, SmartEmbed, ETH2Vec.
    Features are extracted ONCE before the loop.
    """
    print(f"\n{'='*62}")
    print(f"  {model_name.upper()}")
    print(f"{'='*62}")

    print("  Extracting features ...", flush=True)
    t_feat = time.time()
    X = build_features_fn(texts)
    print(f"  Features ready in {time.time()-t_feat:.1f}s  shape={getattr(X, 'shape', '?')}")

    y_arr  = np.array(labels)
    src_arr = np.array(source_datasets)
    fold_results: List[dict] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(splits):
        # Validate no group leakage (sanity check)
        print(f"\n  Fold {fold_idx+1}/{N_SPLITS}  "
              f"(train={len(tr_idx)}, test={len(te_idx)}) ...", flush=True)

        y_train = y_arr[tr_idx].tolist()
        y_test  = y_arr[te_idx].tolist()
        cw      = get_class_weights(y_train)

        # Index into features
        if hasattr(X, "toarray"):               # sparse matrix
            X_tr = X[tr_idx]
            X_te = X[te_idx]
        else:                                   # dense array
            X_tr = X[tr_idx]
            X_te = X[te_idx]

        t0  = time.time()
        clf = train_model_fn(X_tr, y_train, cw)
        t_train = time.time() - t0

        y_pred = predict_fn(clf, X_te)

        metrics = compute_metrics(y_test, y_pred, src_arr[te_idx])
        metrics.update({
            "fold":         fold_idx + 1,
            "train_time_s": round(t_train, 2),
            "n_train":      len(tr_idx),
            "n_test":       len(te_idx),
            "test_idx":     [int(i) for i in te_idx],
        })
        fold_results.append(metrics)

        print(f"    macro-F1={metrics['macro_f1']:.4f}  "
              f"acc={metrics['accuracy']:.4f}  "
              f"time={t_train:.1f}s")

    agg = aggregate_folds(fold_results)
    out = save_results(model_name, fold_results, agg)

    print(f"\n  {model_name}: macro-F1 = "
          f"{agg['macro_f1_mean']:.4f} +/- {agg['macro_f1_std']:.4f}")
    print(f"  Saved -> {out}")
    return {"model": model_name, "folds": fold_results, "aggregate": agg}
