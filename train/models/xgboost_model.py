"""train/models/xgboost_model.py — XGBoost baseline with Fused Features."""
import numpy as np
from typing import List, Tuple

from xgboost import XGBClassifier

from train.config import NUM_LABELS, get_seed
from train.evaluate import run_sklearn_cv
from train.features.fused_features import fused_features, apply_gaussian_augmentation


def run(
    texts: List[str],
    labels: List[int],
    source_datasets: List[str],
    splits: List[Tuple[np.ndarray, np.ndarray]],
    persist: bool = True,
    hp: dict | None = None,
) -> dict:
    hp = {**dict(n_estimators=200, max_depth=6, learning_rate=0.1), **(hp or {})}

    # ── Extract Fused Features ───────────────────────────────────────────
    X_all = fused_features(texts, device='cuda' if __import__('torch').cuda.is_available() else 'cpu')

    def train_fn(X_tr, y_tr, sw, tr_idx):
        # Gaussian-noise augmentation of the training split only; every noisy copy
        # inherits the weight of its original (class- and, per profile, source-balanced).
        X_tr_aug, y_tr_aug, sample_weights = apply_gaussian_augmentation(
            X_tr, y_tr, num_classes=NUM_LABELS, weights=sw)

        clf = XGBClassifier(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"],
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=get_seed()
        )
        clf.fit(X_tr_aug, y_tr_aug, sample_weight=sample_weights)
        
        return clf

    def predict_fn(clf, X_te):
        return clf.predict(X_te).tolist()

    return run_sklearn_cv(
        model_name      = "xgboost",
        build_features_fn = lambda _: X_all,
        train_model_fn  = train_fn,
        predict_fn      = predict_fn,
        texts           = texts,
        labels          = labels,
        source_datasets = source_datasets,
        splits          = splits,
        persist         = persist,
    )
