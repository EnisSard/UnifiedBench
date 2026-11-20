"""
train/models/smartembed_model.py
--------------------------------
SmartEmbed-style model (after Gao et al., ICSME 2019), adapted to multi-class classification.

Pipeline (per cross-validation fold, fit on the training split only):
  1. Normalise Solidity source -> structural AST-style token stream
     (identifiers -> VAR, literals -> LIT, keywords kept verbatim).
  2. Train a Skip-Gram Word2Vec model (d=100, window=5, 10 epochs) on the
     training contracts' token streams.
  3. Represent every contract as the mean pool of its token embeddings.
  4. Classify with a class-balanced Random Forest (100 trees).
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from train.config import SMARTEMBED_PARAMS, get_seed, warn_if_hashseed_unset
from train.features.ast_features import _normalize, build_word2vec, embed
from train.models._real_cv import run_real_cv


def _fit_predict(train_texts, y_train, train_sources, test_texts, sample_weights=None, hp=None):
    hp = {**dict(n_estimators=100, max_features="sqrt"), **(hp or {})}
    tr_streams = [_normalize(t) for t in train_texts]
    te_streams = [_normalize(t) for t in test_texts]

    w2v = build_word2vec(tr_streams)                       # fit on TRAIN only
    X_tr = np.vstack([embed(s, w2v) for s in tr_streams])
    X_te = np.vstack([embed(s, w2v) for s in te_streams])

    clf = RandomForestClassifier(
        n_estimators=hp["n_estimators"],
        max_features=hp["max_features"],
        random_state=get_seed(),
        class_weight=None,   # class (and source) balance comes from sample_weights
        n_jobs=-1,
    )
    clf.fit(X_tr, y_train, sample_weight=sample_weights)
    return clf.predict(X_te).tolist()


def run(texts, labels, source_datasets, splits, persist=True, hp=None):
    warn_if_hashseed_unset()
    print(f"  [SmartEmbed] Word2Vec Skip-Gram d={SMARTEMBED_PARAMS['vector_size']} "
          f"+ Random Forest (100 trees)")
    return run_real_cv(
        "smartembed", texts, labels, source_datasets, splits,
        fit_predict_fn=lambda *a, **k: _fit_predict(*a, **k, hp=hp), persist=persist,
    )
