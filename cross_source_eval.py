"""
cross_source_eval.py
--------------------
Leave-one-source-out (LOSO) transfer: train on all SolidiFI contracts and test on
all real-world contracts, and the reverse. Each direction is run for every seed in
train.config.LOSO_SEEDS so the stochastic models' seed-to-seed variability can be
reported. Per-seed predictions are kept for the statistics in analyze_results.py:

    results/loso/seed_<s>/<train>_to_<test>/<model>/fold_results.json
    results/loso/loso_summary.json      macro-F1 per model / direction / seed

Run with PYTHONHASHSEED=0 (see README).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from train.config import LOSO_SEEDS, RESULTS_DIR, set_seed, set_results_dir
from train.data_loader import load_records
from train.models import (
    xgboost_model, smartembed_model, eth2vec_model, transformer_model,
    deepsolid_model, hybridvuldetect_model, gnn_model,
)

MODELS = ["xgboost", "smartembed", "eth2vec", "deepsolid", "hybridvuldetect", "gnn", "smartbert", "codebert"]
SOURCES = ["solidifi", "realworld"]


def _run(name, texts, labels, sources, splits):
    if name in ("smartbert", "codebert"):
        return transformer_model.run(name, texts, labels, sources, splits)
    module = {
        "xgboost": xgboost_model, "smartembed": smartembed_model, "eth2vec": eth2vec_model,
        "deepsolid": deepsolid_model, "hybridvuldetect": hybridvuldetect_model, "gnn": gnn_model,
    }[name]
    return module.run(texts, labels, sources, splits)


def run_cross_source(seeds, models, out_root: Path, overwrite: bool = False):
    texts, labels, group_ids, source_datasets, filenames = load_records()
    src = np.asarray(source_datasets)
    summary_path = out_root / "loso_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    print("--- Leave-one-source-out evaluation ---")
    for seed in seeds:
        for train_src in SOURCES:
            for test_src in SOURCES:
                if train_src == test_src:
                    continue
                direction = f"{train_src}_to_{test_src}"
                train_idx = np.where(src == train_src)[0]
                test_idx = np.where(src == test_src)[0]
                splits = [(train_idx, test_idx)]
                print(f"\n[LOSO] seed {seed}  {direction}  (train {len(train_idx)}, test {len(test_idx)})")
                for name in models:
                    run_dir = out_root / f"seed_{seed}" / direction
                    done = run_dir / name / "fold_results.json"
                    if done.exists() and not overwrite:
                        f1 = json.loads(done.read_text())["folds"][0]["macro_f1"]
                    else:
                        set_seed(seed)
                        set_results_dir(run_dir)
                        try:
                            f1 = _run(name, texts, labels, source_datasets, splits)["aggregate"]["macro_f1_mean"]
                        except Exception as e:
                            print(f"  {name} failed: {e}")
                            continue
                    print(f"  {name:<16} macro-F1 = {f1:.4f}")
                    summary.setdefault(name, {}).setdefault(direction, {})[str(seed)] = f1
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[OK] summary -> {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Leave-one-source-out transfer evaluation.")
    ap.add_argument("--seeds", nargs="+", type=int, default=LOSO_SEEDS)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    ap.add_argument("--out", default=str(RESULTS_DIR / "loso"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    run_cross_source(a.seeds, a.models, Path(a.out), a.overwrite)
