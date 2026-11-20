"""Shows how many folds are finished per stage and model.   python progress.py results/runA"""
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/runA")
expected = {"repeats": (3 if root.name == "runB" else 10) * 5, "seed_study": 4 * 5,
            "ablation_singleton_groups": 5, "loso": 5 * 2}
for stage in ["repeats", "loso", "seed_study", "ablation_singleton_groups"]:
    d = root / stage
    counts = defaultdict(int)
    for f in d.rglob("folds/fold_*.json"):
        counts[f.parent.parent.name] += 1
    if not counts:
        print(f"{stage:<28} not started")
        continue
    print(f"{stage:<28} " + "  ".join(f"{m} {n}/{expected[stage]}" for m, n in sorted(counts.items())))
