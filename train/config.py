"""train/config.py — Central configuration for all models."""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR     = PROJECT_ROOT / "data"
EXPANDED_DIR  = DATA_DIR / "expanded_real_world_set"
SYNTHETIC_DIR = DATA_DIR / "synthetic_benchmark"
MANIFEST     = EXPANDED_DIR / "reproducibility_manifest.csv"
SYNTHETIC_MANIFEST = SYNTHETIC_DIR / "synthetic_manifest.csv"
RESULTS_DIR  = PROJECT_ROOT / "results"
CACHE_DIR    = DATA_DIR / ".cache"          # compiled opcodes, etc.

# ── Labels ────────────────────────────────────────────────
LABELS = [
    "access_control",
    "arithmetic",
    "front_running",
    "reentrancy",
    "time_manipulation",
    "unchecked_low_level_calls",
]
LABEL2IDX  = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL  = {i: l for i, l in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# ── CV settings ───────────────────────────────────────────
N_SPLITS    = 5
RANDOM_SEED = 42            # default seed; repeated runs override it via set_seed()

# Repeated grouped cross-validation: repeat r uses seed REPEAT_SEED_BASE + r for
# BOTH the fold split and every model's stochastic components.
N_REPEATS        = 10
REPEAT_SEED_BASE = 42
LOSO_SEEDS       = [42, 43, 44, 45, 46]

# ── Run profile (set with the environment variable UB_PROFILE) ─────────────
# base  fixes only: comments stripped from inputs, MLP validation split before
#       augmentation and by clone family, 593-contract dataset
# runA  base + Transformers read the whole contract in 512-token chunks (MiniLM
#       likewise in 256-token chunks), SolidiFI weakness-named identifiers renamed
#       to neutral names, and source-balanced sample weights (SolidiFI and real-world
#       contracts carry equal total weight in every training split)
# runB  runA + nested-CV hyperparameter tuning + multi-label CGT contracts +
#       risk-ranked chunk selection (long contracts keep the chunks with the most
#       security-relevant operations instead of evenly spaced ones)
import os as _os

_PROFILES = {
    "base": dict(chunking=False, neutralize_solidifi=False, source_balance=False, tune=False, multilabel=False, chunk_select="even"),
    "runA": dict(chunking=True,  neutralize_solidifi=True,  source_balance=True,  tune=False, multilabel=False, chunk_select="even"),
    "runB": dict(chunking=True,  neutralize_solidifi=True,  source_balance=True,  tune=True,  multilabel=True,  chunk_select="risk"),
}
_PROFILE_NAME = _os.environ.get("UB_PROFILE", "base")
if _PROFILE_NAME not in _PROFILES:
    raise ValueError(f"UB_PROFILE must be one of {list(_PROFILES)}, got {_PROFILE_NAME!r}")
PROFILE = dict(name=_PROFILE_NAME, **_PROFILES[_PROFILE_NAME])

# Run B builds its dataset (with the multi-label CGT contracts) into its own folder,
# so it never overwrites the 593-contract benchmark used by the other profiles.
if PROFILE["multilabel"]:
    EXPANDED_DIR = DATA_DIR / "expanded_real_world_set_multilabel"
    MANIFEST = EXPANDED_DIR / "reproducibility_manifest.csv"
    SYNTHETIC_DIR = DATA_DIR / "synthetic_benchmark_multilabel"   # same files, group ids
    SYNTHETIC_MANIFEST = SYNTHETIC_DIR / "synthetic_manifest.csv"  # of the larger pool

# Transformer chunking: at most this many 512-token chunks per contract, spread
# evenly over the contract when it is longer.
MAX_CHUNKS = 6
# MiniLM chunking for the fused vector: at most this many 256-token chunks.
MINILM_MAX_CHUNKS = 16

# ── Runtime state (seed + output folder of the current run) ──
_RUNTIME = {"seed": RANDOM_SEED, "results_dir": RESULTS_DIR, "groups": None}


def set_groups(group_ids) -> None:
    """Clone-family id of every contract (used for group-aware validation splits)."""
    _RUNTIME["groups"] = list(group_ids)


def get_groups():
    return _RUNTIME["groups"]


# Multi-label rows (profile runB). A contract with label set {A, B} is loaded as one
# row per label; every row carries the contract key, the full label set and the
# factor 1/|set|, so the contract's total training weight stays that of one contract.
# Scoring collapses the rows back to one prediction per contract (see evaluate.py).
def set_rows(keys, label_sets) -> None:
    _RUNTIME["rows"] = dict(keys=list(keys), label_sets=[frozenset(s) for s in label_sets])


def get_rows():
    return _RUNTIME.get("rows")


STOP_FILE = PROJECT_ROOT / "STOP"


class StopRequested(Exception):
    """Raised between folds when the STOP file exists."""


def get_seed() -> int:
    """Seed of the current run (split + model initialisation + augmentation)."""
    return _RUNTIME["seed"]


def set_seed(seed: int) -> None:
    _RUNTIME["seed"] = int(seed)


def get_results_dir() -> Path:
    """Folder that receives <model>/fold_results.json for the current run."""
    return _RUNTIME["results_dir"]


def set_results_dir(path) -> None:
    _RUNTIME["results_dir"] = Path(path)

# ── XGBoost ───────────────────────────────────────────────
XGBOOST_PARAMS = dict(
    n_estimators    = 200,
    max_depth       = 6,
    learning_rate   = 0.1,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    eval_metric     = "mlogloss",
    random_state    = RANDOM_SEED,
    n_jobs          = -1,
)

# ── SmartEmbed (Word2Vec) ─────────────────────────────────
SMARTEMBED_PARAMS = dict(
    vector_size = 100,
    window      = 5,
    min_count   = 1,
    sg          = 1,        # Skip-Gram (as per paper)
    epochs      = 10,
    workers     = 1,        # single thread: multi-threaded gensim training is not repeatable
    seed        = RANDOM_SEED,
)

# ── ETH2Vec (Doc2Vec PV-DM) ──────────────────────────────
ETH2VEC_PARAMS = dict(
    vector_size = 100,
    window      = 5,
    min_count   = 1,
    dm          = 1,        # PV-DM mode (as per paper)
    epochs      = 20,
    workers     = 1,        # single thread: multi-threaded gensim training is not repeatable
    seed        = RANDOM_SEED,
)
# ETH2Vec compiles each contract with the closest solc from the matrix in
# train/features/opcode_features.py (0.4.11 - 0.8.20).


# ── Repeatability of the gensim models (SmartEmbed, ETH2Vec) ──────────────
def warn_if_hashseed_unset() -> None:
    """gensim derives its random initial vectors from Python's str hash, which changes
    every process unless PYTHONHASHSEED is fixed. Warn once if it is not."""
    import os
    if os.environ.get("PYTHONHASHSEED") is None:
        print("  [WARN] PYTHONHASHSEED is not set: SmartEmbed / ETH2Vec will not be exactly "
              "repeatable. Run with PYTHONHASHSEED=0 (see README).")
