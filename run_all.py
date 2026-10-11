"""
run_all.py
----------
Orchestrates the 8-model training pipeline under repeated Stratified Grouped
5-fold cross-validation.

Repeat r uses seed REPEAT_SEED_BASE + r (train/config.py) for the fold split AND
for every stochastic component of every model (weight initialisation, data-loader
shuffling, dropout, Word2Vec/Doc2Vec initialisation, tree sub-sampling, Gaussian
augmentation). All models of a repeat share the same folds, so per-fold results are
paired across models. Outputs:

    results/repeats/seed_<s>/<model>/fold_results.json

Models:
    xgboost         - XGBoost on the 406-d fused vector
    smartembed      - Word2Vec (skip-gram) + Random Forest
    eth2vec         - EVM opcode Doc2Vec (PV-DM) + RBF-SVM
    deepsolid       - dual-branch MLP on the fused vector
    hybridvuldetect - dual-activation MLP on the fused vector
    codebert        - fine-tuned microsoft/codebert-base
    smartbert       - fine-tuned web3se/SmartBERT-v3
    gnn             - Contract-GCN (graph convolutional network)

Examples:
    python run_all.py                       # 10 repeats (seeds 42..51), all models
    python run_all.py --repeats 1           # single run, seed 42
    python run_all.py --seeds 42 43 --models xgboost gnn
    python run_all.py --seeds 43 44 45 46 --split-seed 42 --out results/seed_study
                                            # model-seed study: folds fixed at seed 42,
                                            # only the training seed changes
Already finished (seed, model) pairs are skipped unless --overwrite is given.
Run with PYTHONHASHSEED=0 so the gensim models are repeatable (see README).
"""

import argparse
import time
import traceback

from train.config import (
    N_REPEATS, REPEAT_SEED_BASE, RESULTS_DIR, set_seed, set_results_dir,
)
from train.data_loader import load_records
from train.evaluate import get_cv_splits

from train.models import xgboost_model, smartembed_model, eth2vec_model
from train.models import transformer_model, deepsolid_model, hybridvuldetect_model, gnn_model

ALL_MODELS = ["xgboost", "smartembed", "eth2vec", "smartbert", "codebert", "deepsolid", "hybridvuldetect", "gnn"]
TRANSFORMER_MODELS = {"smartbert", "codebert"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run smart contract vulnerability detection models.")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, help="Run only these models")
    parser.add_argument("--skip", nargs="+", choices=ALL_MODELS, default=[], help="Skip these models")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help=f"number of CV repeats (default {N_REPEATS}); seeds start at {REPEAT_SEED_BASE}")
    parser.add_argument("--seeds", nargs="+", type=int, help="explicit list of repeat seeds (overrides --repeats)")
    parser.add_argument("--out", default=str(RESULTS_DIR / "repeats"), help="output root folder")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="keep the fold split fixed at this seed and vary only the model seed")
    parser.add_argument("--file-level-solidifi-groups", action="store_true",
                        help="ABLATION ONLY: one group per SolidiFI file (file-level grouping), "
                             "to measure how much the base-contract grouping changes the results")
    parser.add_argument("--overwrite", action="store_true", help="re-run (seed, model) pairs that already have results")
    return parser.parse_args()


def run_model(model_name, texts, labels, source_datasets, splits):
    if model_name == "xgboost":
        return xgboost_model.run(texts, labels, source_datasets, splits)
    if model_name == "smartembed":
        return smartembed_model.run(texts, labels, source_datasets, splits)
    if model_name == "eth2vec":
        return eth2vec_model.run(texts, labels, source_datasets, splits)
    if model_name == "deepsolid":
        return deepsolid_model.run(texts, labels, source_datasets, splits)
    if model_name == "hybridvuldetect":
        return hybridvuldetect_model.run(texts, labels, source_datasets, splits)
    if model_name == "gnn":
        return gnn_model.run(texts, labels, source_datasets, splits)
    if model_name in TRANSFORMER_MODELS:
        return transformer_model.run(model_name, texts, labels, source_datasets, splits)
    raise ValueError(model_name)


def main():
    from pathlib import Path
    args = parse_args()
    models_to_run = args.models if args.models else ALL_MODELS
    models_to_run = [m for m in models_to_run if m not in (args.skip or [])]
    seeds = args.seeds if args.seeds else [REPEAT_SEED_BASE + r for r in range(args.repeats)]
    out_root = Path(args.out)

    print("=" * 62)
    print("  Smart Contract Vulnerability Detection - Training Pipeline")
    print("=" * 62)
    print(f"\n  Models to run : {models_to_run}")
    print(f"  Repeat seeds  : {seeds}")
    if args.split_seed is not None:
        print(f"  Split seed    : {args.split_seed} (fixed; only the model seed changes)")
    print(f"  Output        : {out_root}")

    print("\n[LOAD] Reading unified dataset ...")
    texts, labels, group_ids, source_datasets, filenames = load_records()
    if args.file_level_solidifi_groups:
        group_ids = [f if s == "solidifi" else g for f, s, g in zip(filenames, source_datasets, group_ids)]
        print(f"  [ABLATION] SolidiFI files are singleton groups -> {len(set(group_ids))} groups")

    t_total = time.time()
    failures = []
    for seed in seeds:
        set_seed(seed)
        set_results_dir(out_root / f"seed_{seed}")
        split_seed = seed if args.split_seed is None else args.split_seed
        print(f"\n[SPLIT] seed {split_seed}: StratifiedGroupKFold splits ...")
        splits = get_cv_splits(labels, group_ids, seed=split_seed)
        print("        test-fold sizes:", [len(te) for _, te in splits])

        for model_name in models_to_run:
            done = out_root / f"seed_{seed}" / model_name / "fold_results.json"
            if done.exists() and not args.overwrite:
                print(f"  [{model_name}] seed {seed}: already done, skipping")
                continue
            set_seed(seed)          # every model starts from the same run seed
            t_model = time.time()
            try:
                run_model(model_name, texts, labels, source_datasets, splits)
                print(f"  [{model_name}] seed {seed}: done in {time.time() - t_model:.0f}s", flush=True)
            except Exception as e:
                failures.append((seed, model_name, str(e)))
                print(f"\n  [ERROR] {model_name} (seed {seed}) failed: {e}")
                traceback.print_exc()

    total_elapsed = time.time() - t_total
    print(f"\n{'='*62}")
    print(f"  All done in {total_elapsed/60:.1f} min.  Failures: {failures or 'none'}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
