"""
run_queue.py
------------
Runs every experiment of one profile in order. Each stage is resumable: finished
folds are checkpointed, so rerunning the same command continues where it stopped.

    python run_queue.py runA            # main CV, LOSO, seed study, grouping ablation
    python run_queue.py runA --only main

Pause:  create a file named STOP in this folder (or press Ctrl+C). The fold that is
        running is discarded; every finished fold is kept.
Resume: delete STOP and run the same command again.
Progress: python progress.py results/runA
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
NEURAL = ["deepsolid", "hybridvuldetect", "codebert", "smartbert", "gnn"]


def stages(out: Path, profile: str):
    if profile == "runB":
        # tuned + multi-label: main CV only, 3 repeats (nested tuning is ~4x the cost)
        return [("main", ["run_all.py", "--repeats", "3", "--out", str(out / "repeats")])]
    return [
        ("main", ["run_all.py", "--out", str(out / "repeats")]),
        ("loso", ["cross_source_eval.py", "--out", str(out / "loso")]),
        ("seeds", ["run_all.py", "--seeds", "43", "44", "45", "46", "--split-seed", "42",
                   "--models", *NEURAL, "--out", str(out / "seed_study")]),
        ("ablation", ["run_all.py", "--repeats", "1", "--file-level-solidifi-groups",
                      "--out", str(out / "ablation_singleton_groups")]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", choices=["base", "runA", "runB"])
    ap.add_argument("--only", nargs="+", help="run only these stages")
    a = ap.parse_args()
    out = ROOT / "results" / a.profile
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, UB_PROFILE=a.profile, PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    for name, cmd in stages(out, a.profile):
        if a.only and name not in a.only:
            continue
        if (ROOT / "STOP").exists():
            print("[QUEUE] STOP file present -- not starting further stages.")
            sys.exit(3)
        print(f"\n######## [{a.profile}] stage {name}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env).returncode
        if rc == 3:
            print("[QUEUE] stopped; rerun the same command to resume.")
            sys.exit(3)
        if rc != 0:
            print(f"[QUEUE] stage {name} exited with {rc}; continuing with the next stage.")
    print(f"\n[QUEUE] {a.profile}: all stages finished.")


if __name__ == "__main__":
    main()
