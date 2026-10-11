"""
setup_data.py
-------------
Checks for the /data folder in the project directory and clones the three
source corpora that UnifiedBench is built from (skipping any already present):
  - SolidiFI-benchmark  (DependableSystemsLab)  synthetic, bug-injected contracts
  - smartbugs-curated   (smartbugs)              manually annotated contracts
  - CGT                 (gsalzer)                Consolidated Ground Truth

Each corpus is fetched at the commit it was pinned to. Run build_unified_dataset.py next.
"""

import shutil
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"

REPOS = {
    # Commits pinned to the exact versions UnifiedBench was built from.
    "SolidiFI-benchmark": {
        "url": "https://github.com/DependableSystemsLab/SolidiFI-benchmark",
        "dest": DATA_DIR / "SolidiFI-benchmark",
        "commit": "4b0573e1b3f7031396de6f48f7f3e7380222ad3a",
    },
    "smartbugs-curated": {
        "url": "https://github.com/smartbugs/smartbugs-curated",
        "dest": DATA_DIR / "smartbugs-curated",
        "commit": "230e649123477eff332742a59a1c7cc6dc286cab",
    },
    "CGT": {
        "url": "https://github.com/gsalzer/cgt.git",
        "dest": DATA_DIR / "CGT",
        "commit": "f8cd72cf7fbbfebc809c454667eee271706a4b2b",
    },
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def run(cmd: list, cwd=None) -> None:
    """Run a shell command and stream output to the terminal."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def git_clone(url: str, dest: Path, commit: str) -> None:
    """Fetch exactly `commit` of `url` into dest (shallow). core.longpaths is enabled
    because some CGT file paths exceed the default Windows path-length limit."""
    print(f"\n[CLONE] {url} @ {commit[:7]}")
    print(f"        -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    run(["git", "init", "-q"], cwd=dest)
    run(["git", "config", "core.longpaths", "true"], cwd=dest)
    run(["git", "config", "core.autocrlf", "false"], cwd=dest)   # identical bytes on every OS
    run(["git", "remote", "add", "origin", url], cwd=dest)
    run(["git", "fetch", "-q", "--depth", "1", "origin", commit], cwd=dest)
    run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest)


# ──────────────────────────────────────────────
# Step 1 – Ensure /data and both repos exist
# ──────────────────────────────────────────────

def ensure_data() -> None:
    if DATA_DIR.exists():
        print(f"[OK] /data folder already exists: {DATA_DIR}")
    else:
        print("[INFO] /data folder not found. Creating it ...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, cfg in REPOS.items():
        dest: Path = cfg["dest"]
        if dest.exists():
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, text=True,
                                  capture_output=True).stdout.strip()
            if head == cfg["commit"]:
                print(f"[OK] '{name}' already present at the pinned commit: {dest}")
            else:
                print(f"[WARN] '{name}' already present but not at the pinned commit "
                      f"{cfg['commit'][:7]} (found {head[:7] or 'no git checkout'}); delete {dest} "
                      f"and re-run to download the exact version used in the paper")
        else:
            try:
                git_clone(cfg["url"], dest, cfg["commit"])
            except BaseException:
                shutil.rmtree(dest, ignore_errors=True)    # never leave a half-downloaded corpus
                raise
            print(f"[DONE] Cloned '{name}'")


def main() -> None:
    ensure_data()
    print("\nDone. Next: python build_unified_dataset.py")


if __name__ == "__main__":
    main()
