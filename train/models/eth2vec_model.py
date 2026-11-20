"""
train/models/eth2vec_model.py
-----------------------------
ETH2Vec-style model (after Ashizawa et al., BSCI 2021), adapted to multi-class classification.

Pipeline (per cross-validation fold, fit on the training split only):
  1. Compile every contract to EVM runtime bytecode and disassemble it to an
     opcode-mnemonic sequence (see train/features/opcode_features.py).
  2. Train a PV-DM Doc2Vec model (d=100, window=5, 20 epochs) on the training
     contracts' opcode sequences; infer 100-d vectors for the training AND test
     contracts with the trained model (same procedure on both sides).
  3. Classify with a class-balanced RBF-kernel SVM (C=1.0).

ETH2Vec is a *bytecode* method: it is trained only on the contracts that compile
with the solc version selected for them (the closest to the declared pragma from the
matrix 0.4.11 - 0.8.20; 0.4.24 when no pragma is declared). Non-compiling TEST contracts
are still scored -- they receive the fold's majority training class as a fallback
prediction -- so ETH2Vec is evaluated on exactly the same contracts as every other
model; ``test_eligible`` in fold_results.json marks the really classified ones.
"""
import json

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.svm import SVC

from train.config import ETH2VEC_PARAMS, get_seed, get_results_dir, warn_if_hashseed_unset
from train.features.opcode_features import build_opcode_sequences
from train.models._real_cv import run_real_cv

# opcode sequences are built once (cached) and indexed by the fold driver
_SEQS = None
_STATUS = None


def _prep(texts):
    global _SEQS, _STATUS
    seqs, status, stats = build_opcode_sequences(list(texts))
    _SEQS = {t: s for t, s in zip(texts, seqs)}
    _STATUS = {t: st for t, st in zip(texts, status)}
    return stats


def _fit_predict(train_texts, y_train, train_sources, test_texts, sample_weights=None, hp=None):
    hp = {**dict(C=1.0), **(hp or {})}
    tr_docs = [
        TaggedDocument(_SEQS[t], [i]) for i, t in enumerate(train_texts)
    ]
    d2v = Doc2Vec(
        documents=tr_docs,
        vector_size=ETH2VEC_PARAMS["vector_size"],
        window=ETH2VEC_PARAMS["window"],
        min_count=ETH2VEC_PARAMS["min_count"],
        dm=ETH2VEC_PARAMS["dm"],
        epochs=ETH2VEC_PARAMS["epochs"],
        workers=ETH2VEC_PARAMS["workers"],
        seed=get_seed(),
    )
    # Train and test vectors must come from the SAME procedure. The jointly-trained
    # paragraph vectors (d2v.dv) live in a different distribution than vectors produced
    # by infer_vector, so an SVM fitted on d2v.dv and applied to inferred test vectors
    # degrades to near chance (measured: 78% training accuracy on d2v.dv vs 37% on
    # re-inferred vectors of the SAME training contracts). Both sides are therefore
    # re-inferred with the trained model.
    infer = lambda t: d2v.infer_vector(_SEQS[t], epochs=ETH2VEC_PARAMS["epochs"])
    X_tr = np.vstack([infer(t) for t in train_texts])
    X_te = np.vstack([infer(t) for t in test_texts])

    clf = SVC(kernel="rbf", C=hp["C"], random_state=get_seed())  # balance via sample_weights
    clf.fit(X_tr, y_train, sample_weight=sample_weights)
    return clf.predict(X_te).tolist()


def run(texts, labels, source_datasets, splits, persist=True, hp=None):
    warn_if_hashseed_unset()
    print("  [ETH2Vec] Doc2Vec PV-DM d=100 + RBF-SVM  (bytecode / opcode method)")
    _prep(list(texts))
    eligible = np.array([_STATUS[t] == "compiled" for t in texts], dtype=bool)

    if persist:
        get_results_dir().joinpath("eth2vec").mkdir(parents=True, exist_ok=True)
        with get_results_dir().joinpath("eth2vec", "compilation_stats.json").open("w") as f:
            json.dump(
                {
                    "total": len(texts),
                    "compiled": int(eligible.sum()),
                    "fallback": int((~eligible).sum()),
                    "compiled_by_source": {
                        s: int(np.sum(eligible & (np.asarray(source_datasets) == s)))
                        for s in sorted(set(source_datasets))
                    },
                },
                f,
                indent=2,
            )

    return run_real_cv(
        "eth2vec", texts, labels, source_datasets, splits,
        fit_predict_fn=lambda *a, **k: _fit_predict(*a, **k, hp=hp), eligible_mask=eligible, persist=persist,
    )
