"""
train/tuning.py
---------------
Nested cross-validation hyperparameter tuning (profile runB).

For every outer fold, the outer training split is divided once more into an inner
training part and an inner validation part (one StratifiedGroupKFold(5) split, so no
clone family crosses it). Every configuration of the model's small grid is trained on
the inner part and scored by macro-F1 on the REAL-WORLD contracts of the inner
validation part (the quantity the benchmark cares about). The best configuration is
then refitted on the whole outer training split and evaluated once on the outer test
split, which is never seen during selection. Ties go to the earlier configuration;
the first configuration of every grid is the untuned default of profile runA.

Grids are deliberately small (3 configurations) and follow common practice for each
model family (e.g. BERT fine-tuning: lr in {1e-5, 2e-5, 3e-5}, 5-8 epochs).

Resumable: every inner score is saved to <model>/inner/fold_<k>_cfg_<j>.json and the
outer fold to the usual fold checkpoint, so a stopped run loses at most one training.
"""
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from train.config import get_groups, get_results_dir, get_seed
from train.evaluate import check_stop, run_folds

GRIDS = {
    "xgboost": [dict(n_estimators=200, max_depth=6, learning_rate=0.1),
                dict(n_estimators=400, max_depth=4, learning_rate=0.05),
                dict(n_estimators=300, max_depth=8, learning_rate=0.1)],
    "smartembed": [dict(n_estimators=100, max_features="sqrt"),
                   dict(n_estimators=300, max_features="sqrt"),
                   dict(n_estimators=300, max_features=0.3)],
    "eth2vec": [dict(C=1.0), dict(C=10.0), dict(C=100.0)],
    "deepsolid": [dict(lr=5e-4, hidden=128, dropout=0.5),
                  dict(lr=1e-3, hidden=256, dropout=0.5),
                  dict(lr=2e-4, hidden=128, dropout=0.3)],
    "hybridvuldetect": [dict(lr=5e-4, hidden=128, dropout=0.5),
                        dict(lr=1e-3, hidden=256, dropout=0.5),
                        dict(lr=2e-4, hidden=128, dropout=0.3)],
    "gnn": [dict(lr=1e-3, epochs=35, hidden=128, dropout=0.5),
            dict(lr=5e-4, epochs=60, hidden=256, dropout=0.5),
            dict(lr=2e-3, epochs=35, hidden=128, dropout=0.3)],
    "codebert": [dict(lr=2e-5, epochs=5), dict(lr=3e-5, epochs=8), dict(lr=1e-5, epochs=8)],
    "smartbert": [dict(lr=2e-5, epochs=5), dict(lr=3e-5, epochs=8), dict(lr=1e-5, epochs=8)],
}


def inner_split(tr_idx, labels, seed):
    """One grouped, stratified 80/20 split of the outer training indices."""
    tr_idx = np.asarray(tr_idx)
    g = np.asarray(get_groups(), dtype=object)[tr_idx]
    y = np.asarray(labels)[tr_idx]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    a, b = next(sgkf.split(np.zeros(len(tr_idx)), y, g))
    return tr_idx[a], tr_idx[b]


def _score(fold: dict) -> float:
    return float(fold.get("realworld_macro_f1", fold["macro_f1"]))


def run_tuned(model_name, run_fn, texts, labels, source_datasets, splits):
    """run_fn(texts, labels, sources, splits, persist=False, hp=cfg) -> result dict."""
    grid = GRIDS[model_name]
    inner_dir = get_results_dir() / model_name / "inner"

    def fold_fn(k, tr_idx, te_idx):
        in_tr, in_val = inner_split(tr_idx, labels, seed=get_seed() * 100 + k)
        scores = []
        for j, cfg in enumerate(grid):
            f = inner_dir / f"fold_{k}_cfg_{j}.json"
            rec = None
            if f.exists():
                try:
                    rec = json.loads(f.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    rec = None
                if rec and (rec.get("cfg") != cfg or rec.get("val_idx") != in_val.tolist()):
                    rec = None
            if rec is None:
                check_stop()
                print(f"\n  [tune] {model_name} fold {k}: config {j + 1}/{len(grid)} {cfg}", flush=True)
                r = run_fn(texts, labels, source_datasets, [(in_tr, in_val)], persist=False, hp=cfg)
                rec = dict(cfg=cfg, score=_score(r["folds"][0]), val_idx=in_val.tolist())
                inner_dir.mkdir(parents=True, exist_ok=True)
                tmp = f.with_suffix(".tmp")
                tmp.write_text(json.dumps(rec), encoding="utf-8")
                os.replace(tmp, f)
            scores.append(rec["score"])
        best = int(np.argmax(scores))
        print(f"  [tune] {model_name} fold {k}: inner real-world macro-F1 "
              f"{[round(s, 3) for s in scores]} -> config {best + 1} {grid[best]}", flush=True)
        check_stop()
        r = run_fn(texts, labels, source_datasets, [(tr_idx, te_idx)], persist=False, hp=grid[best])
        out = r["folds"][0]
        out.update(tuned_hp=grid[best], inner_scores=scores)
        return out

    return run_folds(model_name, splits, fold_fn, persist=True)
