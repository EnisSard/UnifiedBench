"""
train/models/_real_cv.py
------------------------
Shared cross-validation driver for the models that learn their own representation:
SmartEmbed (Word2Vec+RF), ETH2Vec (Doc2Vec+SVM), and the HuggingFace
Transformers (CodeBERT, SmartBERT).

Every model trains its own representation *inside each fold* (fit on the
training split only, never on the held-out split), then classifies.  Results
are written to ``<results dir>/<model>/fold_results.json`` in the same format as
the other models.
"""
import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from train.evaluate import compute_metrics, get_sample_weights, run_folds


def run_real_cv(
    model_name: str,
    texts: Sequence[str],
    labels: Sequence[int],
    source_datasets: Sequence[str],
    splits: List[Tuple[np.ndarray, np.ndarray]],
    fit_predict_fn: Callable[..., Sequence[int]],
    eligible_mask: Optional[Sequence[bool]] = None,
    persist: bool = True,
) -> dict:
    """
    Parameters
    ----------
    fit_predict_fn(train_texts, y_train, train_sources, test_texts, sample_weights) -> y_pred
        Trains a fresh model on the fold's training data and returns predictions
        for ``test_texts`` (same order/length).
    eligible_mask
        Optional per-contract boolean mask (used by ETH2Vec, which needs EVM
        bytecode). Ineligible contracts are dropped from the TRAINING side of every
        fold. On the TEST side they stay in the evaluation and receive the fallback
        prediction (the majority class of the fold's eligible training contracts),
        so every model is scored on the same contracts; ``test_eligible`` records
        which test contracts were really classified, for matched-subset analyses.
    """
    texts = list(texts)
    y_arr = np.asarray(labels)
    src_arr = np.asarray(source_datasets, dtype=object)
    if eligible_mask is None:
        eligible_mask = np.ones(len(texts), dtype=bool)
    eligible_mask = np.asarray(eligible_mask, dtype=bool)

    print(f"\n{'=' * 62}\n  {model_name.upper()}  (representation fitted per fold)\n{'=' * 62}")
    if not eligible_mask.all():
        print(f"  Eligible contracts: {int(eligible_mask.sum())}/{len(texts)} "
              f"(the rest are excluded from training and get a fallback prediction when tested)")

    def fold_fn(fold_idx, tr_idx, te_idx):
        tr_idx = np.asarray([int(i) for i in tr_idx if eligible_mask[i]], dtype=int)
        te_idx = np.asarray([int(i) for i in te_idx], dtype=int)
        te_ok = eligible_mask[te_idx]

        tr_texts = [texts[i] for i in tr_idx]
        te_texts = [texts[i] for i in te_idx[te_ok]]
        y_tr = y_arr[tr_idx].tolist()
        y_te = y_arr[te_idx].tolist()
        tr_src = src_arr[tr_idx].tolist()
        te_src = src_arr[te_idx].tolist()
        sw = get_sample_weights(y_tr, tr_src, tr_idx)

        print(f"\n  Fold {fold_idx}/{len(splits)}  "
              f"(train={len(tr_idx)}, test={len(te_idx)}) ...", flush=True)

        t0 = time.time()
        pred_ok = list(fit_predict_fn(tr_texts, y_tr, tr_src, te_texts, sw))
        dt = time.time() - t0

        assert len(pred_ok) == int(te_ok.sum()), (len(pred_ok), int(te_ok.sum()))
        fallback = int(np.bincount(np.asarray(y_tr), minlength=1).argmax())
        y_pred = np.full(len(te_idx), fallback, dtype=int)
        y_pred[te_ok] = pred_ok
        metrics = compute_metrics(y_te, y_pred.tolist(), np.asarray(te_src, dtype=object), idx=te_idx)
        metrics.update(
            train_time_s=round(dt, 2),
            n_train=int(len(tr_idx)),
            n_test=int(len(te_idx)),
            test_idx=te_idx.tolist(),
            test_eligible=te_ok.tolist(),
        )
        print(f"    macro-F1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}  "
              f"mcc={metrics['mcc']:.4f}  ({dt:.1f}s)", flush=True)
        return metrics

    return run_folds(model_name, splits, fold_fn, persist=persist)
