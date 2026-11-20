"""train/evaluate.py — Shared CV splits, metrics, and result persistence."""
import json
import os
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
    PROFILE, STOP_FILE, StopRequested, get_rows,
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


def _row_factor(idx) -> np.ndarray | None:
    """1/|label set| per row when multi-label rows are loaded (profile runB), else None."""
    rows = get_rows()
    if idx is None or rows is None or not PROFILE["multilabel"]:
        return None
    return np.array([1.0 / len(rows["label_sets"][int(i)]) for i in idx])


def get_sample_weights(y_train, sources_train, idx=None) -> np.ndarray:
    """Per-sample training weights: the class-balanced weight of each sample's class,
    times (profile runA/runB) a source-balancing factor that gives SolidiFI and
    real-world contracts equal total weight. Normalised to mean 1.
    With multi-label rows (runB, pass the training row indices as `idx`) each row of a
    contract with k labels counts 1/k, and class and source totals use these counts."""
    y = np.asarray(y_train)
    f = _row_factor(idx)
    if f is None:
        cw = get_class_weights(y.tolist())
        w = cw[y].astype(float)
        base = np.ones(len(y))
    else:
        tot = np.bincount(y, weights=f, minlength=NUM_LABELS)
        present = tot > 0
        cw = np.ones(NUM_LABELS)
        cw[present] = f.sum() / (present.sum() * tot[present])
        w = cw[y] * f
        base = f
    if PROFILE["source_balance"]:
        src = np.asarray(sources_train, dtype=object)
        present_s = np.unique(src)
        for s in present_s:
            m = src == s
            w[m] *= base.sum() / (len(present_s) * base[m].sum())
    return w / w.mean()


# ── Fold checkpoints (pause / resume) ─────────────────────────────────────────
# Every finished fold is written to <results dir>/<model>/folds/fold_<k>.json
# (atomically). A restarted run reuses a fold only if its stored test indices match
# the current split; the interrupted fold is recomputed from scratch. Creating the
# file STOP in the project root stops the run cleanly after the current fold.

def _fold_path(model_name: str, k: int) -> Path:
    return get_results_dir() / model_name / "folds" / f"fold_{k}.json"


def load_fold(model_name: str, k: int, te_idx) -> dict | None:
    p = _fold_path(model_name, k)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if rec.get("test_idx") != [int(i) for i in te_idx]:
        print(f"    [ckpt] fold {k}: stored split differs from the current one -> recomputing")
        return None
    return rec


def save_fold(model_name: str, k: int, rec: dict) -> None:
    p = _fold_path(model_name, k)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, p)


def check_stop() -> None:
    if STOP_FILE.exists():
        raise StopRequested(f"STOP file found ({STOP_FILE}); finished folds are saved.")


def run_folds(model_name: str, splits, fold_fn: Callable[[int, np.ndarray, np.ndarray], dict],
              persist: bool = True) -> dict:
    """Runs fold_fn(k, tr_idx, te_idx) -> metrics for every fold not already
    checkpointed, then aggregates and writes fold_results.json."""
    fold_results: List[dict] = []
    for k, (tr_idx, te_idx) in enumerate(splits, start=1):
        rec = load_fold(model_name, k, te_idx) if persist else None
        if rec is not None:
            print(f"  Fold {k}/{len(splits)}: restored from checkpoint "
                  f"(macro-F1={rec['macro_f1']:.4f})", flush=True)
        else:
            check_stop()
            fold_seed = get_seed()
            _seed_everything(fold_seed * 100 + k)
            rec = fold_fn(k, np.asarray(tr_idx), np.asarray(te_idx))
            rec["fold"] = k
            rec.setdefault("test_idx", [int(i) for i in te_idx])
            if persist:
                save_fold(model_name, k, rec)
        fold_results.append(rec)

    agg = aggregate_folds(fold_results)
    if persist:
        out = save_results(model_name, fold_results, agg)
        print(f"  [OK] {model_name}: macro-F1 = {agg['macro_f1_mean']:.4f} "
              f"+/- {agg['macro_f1_std']:.4f}   -> {out}", flush=True)
    return {"model": model_name, "folds": fold_results, "aggregate": agg}


def _seed_everything(s: int) -> None:
    import random
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    try:
        import torch
        torch.manual_seed(s)
    except ImportError:
        pass


# ── Metrics ───────────────────────────────────────────────────────────────────

def _collapse_rows(y_true, y_pred, source_datasets, idx):
    """Multi-label rows (runB) -> one prediction per contract. The first row of a
    contract represents it; the prediction counts as correct when it is ANY label of
    the contract's set (y_true is then set to the prediction, otherwise to that row's
    label). Returns the collapsed arrays, the representative row indices and a
    multi-label mask."""
    rows = get_rows()
    keys, sets = rows["keys"], rows["label_sets"]
    keep, yt, yp, ml = [], [], [], []
    seen = set()
    for pos, i in enumerate(idx):
        i = int(i)
        if keys[i] in seen:
            continue
        seen.add(keys[i])
        p = int(y_pred[pos])
        keep.append(pos)
        yp.append(p)
        yt.append(p if p in sets[i] else int(y_true[pos]))
        ml.append(len(sets[i]) > 1)
    src = None if source_datasets is None else np.asarray(source_datasets)[keep]
    return np.array(yt), np.array(yp), src, [int(idx[k]) for k in keep], np.array(ml)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    source_datasets: np.ndarray | None = None,
    prefix: str = "",
    idx=None,
) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ml = None
    if idx is not None and PROFILE["multilabel"] and get_rows() is not None:
        y_true, y_pred, source_datasets, eval_idx, ml = _collapse_rows(
            y_true, y_pred, source_datasets, idx)

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

    if ml is not None:
        src_arr = np.array(source_datasets)
        single_rw = (src_arr == "realworld") & ~ml
        if single_rw.sum():
            result[f"{prefix}realworld_single_macro_f1"] = float(
                f1_score(y_true[single_rw], y_pred[single_rw], average="macro", zero_division=0))
        if ml.sum():
            result[f"{prefix}multilabel_anymatch_acc"] = float((y_true[ml] == y_pred[ml]).mean())
        result[f"{prefix}eval_idx"] = eval_idx
        result[f"{prefix}multilabel_mask"] = ml.astype(int).tolist()

    result[f"{prefix}y_true"] = y_true.tolist()
    result[f"{prefix}y_pred"] = y_pred.tolist()
    if source_datasets is not None:
        result[f"{prefix}sources"] = src_arr.tolist()

    return result


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_folds(fold_results: List[dict]) -> dict:
    agg: dict = {}
    skip = {"fold", "n_train", "n_test", "y_true", "y_pred", "sources",
            "test_idx", "test_eligible", "eval_idx", "multilabel_mask"}
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


# ── High-level CV runner for feature-based models ─────────────────────────────

def run_sklearn_cv(
    model_name: str,
    build_features_fn: Callable,      # build_features_fn(texts) -> X
    train_model_fn: Callable,         # train_model_fn(X_tr, y_tr, sample_weights, tr_idx) -> clf
    predict_fn: Callable,             # predict_fn(clf, X_te) -> List[int]
    texts: List[str],
    labels: List[int],
    source_datasets: List[str],
    splits: List[Tuple[np.ndarray, np.ndarray]],
    persist: bool = True,
) -> dict:
    """Generic CV loop for the models on a fixed feature matrix (XGBoost, the MLPs).
    Features are extracted once; every fold is checkpointed (see run_folds)."""
    print(f"\n{'='*62}\n  {model_name.upper()}\n{'='*62}")
    t_feat = time.time()
    X = build_features_fn(texts)
    print(f"  Features ready in {time.time()-t_feat:.1f}s  shape={getattr(X, 'shape', '?')}")

    y_arr = np.array(labels)
    src_arr = np.array(source_datasets)

    def fold_fn(k, tr_idx, te_idx):
        print(f"\n  Fold {k}/{len(splits)}  (train={len(tr_idx)}, test={len(te_idx)}) ...", flush=True)
        sw = get_sample_weights(y_arr[tr_idx], src_arr[tr_idx], tr_idx)
        t0 = time.time()
        clf = train_model_fn(X[tr_idx], y_arr[tr_idx], sw, tr_idx)
        t_train = time.time() - t0
        y_pred = predict_fn(clf, X[te_idx])
        m = compute_metrics(y_arr[te_idx].tolist(), y_pred, src_arr[te_idx], idx=te_idx)
        m.update(train_time_s=round(t_train, 2), n_train=len(tr_idx), n_test=len(te_idx),
                 test_idx=[int(i) for i in te_idx])
        print(f"    macro-F1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}  time={t_train:.1f}s", flush=True)
        return m

    return run_folds(model_name, splits, fold_fn, persist=persist)
