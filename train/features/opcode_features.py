"""
train/features/opcode_features.py
----------------------------------
ETH2Vec-inspired opcode extraction (Ashizawa et al., BSCI 2021).

Pipeline:
  1. Detect the Solidity ``pragma`` and compile the contract to EVM
     runtime bytecode with the closest available ``solc`` (py-solc-x).
  2. Disassemble the bytecode to an opcode mnemonic sequence (pyevmasm).
  3. Contracts that do not compile under the selected toolchain (the closest solc
     to the declared pragma; 0.4.24 when no pragma is declared) are marked
     ``fallback``; ``eth2vec_model.py`` does not train on them and gives them the
     fold's majority training class when they are tested.
  4. Cache every result to ``data/.cache/eth2vec_opcodes.json``.
"""

import json
import re
import hashlib
from typing import Dict, List, Tuple

from tqdm import tqdm

from train.config import CACHE_DIR

OPCODE_CACHE = CACHE_DIR / "eth2vec_opcodes.json"

# solc releases we are willing to install / use, newest first per minor line.
_SOLC_MATRIX = [
    "0.4.11", "0.4.15", "0.4.18", "0.4.19", "0.4.21", "0.4.24", "0.4.25", "0.4.26",
    "0.5.0", "0.5.2", "0.5.8", "0.5.11", "0.5.17",
    "0.6.12", "0.7.6", "0.8.20",
]

_PRAGMA_RE = re.compile(r"pragma\s+solidity\s+([^;]+);")
_VER_RE = re.compile(r"(\d+\.\d+\.\d+)")


# ── Source-level opcode-proxy fallback ───────────────────────────────────────
_SOURCE_PROXIES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\.call\s*[\({]"), "CALL"),
    (re.compile(r"\.delegatecall"), "DELEGATECALL"),
    (re.compile(r"\.transfer\s*\("), "CALL"),
    (re.compile(r"\.send\s*\("), "CALL"),
    (re.compile(r"\bselfdestruct\b"), "SELFDESTRUCT"),
    (re.compile(r"\bsuicide\b"), "SELFDESTRUCT"),
    (re.compile(r"\bkeccak256\b"), "SHA3"),
    (re.compile(r"\bsha256\b"), "SHA256"),
    (re.compile(r"\bblock\.timestamp\b|\bnow\b"), "TIMESTAMP"),
    (re.compile(r"\bblock\.number\b"), "NUMBER"),
    (re.compile(r"\bblock\.difficulty\b"), "DIFFICULTY"),
    (re.compile(r"\bblockhash\b"), "BLOCKHASH"),
    (re.compile(r"\btx\.origin\b"), "ORIGIN"),
    (re.compile(r"\bmsg\.sender\b"), "CALLER"),
    (re.compile(r"\bmsg\.value\b"), "CALLVALUE"),
    (re.compile(r"\brequire\s*\("), "JUMPI"),
    (re.compile(r"\bassert\s*\("), "JUMPI"),
    (re.compile(r"\bif\s*\("), "JUMPI"),
    (re.compile(r"\bfor\s*\("), "JUMP"),
    (re.compile(r"\bwhile\s*\("), "JUMP"),
    (re.compile(r"\bmapping\s*\("), "SSTORE"),
    (re.compile(r"\bstorage\b"), "SSTORE"),
    (re.compile(r"\bmemory\b"), "MSTORE"),
    (re.compile(r"\+\+|--|\+=|-="), "ADD"),
    (re.compile(r"\*=|/="), "MUL"),
    (re.compile(r"\breturn\b"), "RETURN"),
    (re.compile(r"\bthrow\b|\brevert\b"), "REVERT"),
    (re.compile(r"\bnew\s+\w"), "CREATE"),
    (re.compile(r"\bemit\b|\bevent\b"), "LOG1"),
]


def _source_to_opcode_proxy(source: str) -> List[str]:
    tokens: List[str] = []
    for line in source.split("\n"):
        for pattern, opcode in _SOURCE_PROXIES:
            tokens.extend([opcode] * len(pattern.findall(line)))
    return tokens if tokens else ["STOP"]


# ── solc handling ───────────────────────────────────────────────────────────
_installed_cache: List[str] = []


def _available_solc() -> List[str]:
    global _installed_cache
    if _installed_cache:
        return _installed_cache
    import solcx
    from packaging.version import Version
    have = {str(v) for v in solcx.get_installed_solc_versions()}
    for v in _SOLC_MATRIX:
        if v not in have:
            try:
                solcx.install_solc(v)
                have.add(v)
            except Exception as e:
                print(f"  [ETH2Vec] WARNING: could not install solc {v}: {e}")
    # only the documented matrix is used, even if other solc versions are installed
    _installed_cache = sorted(have & set(_SOLC_MATRIX), key=Version)
    return _installed_cache


def _pick_solc(source: str) -> str:
    from packaging.version import Version
    avail = _available_solc()
    m = _PRAGMA_RE.search(source)
    want = None
    if m:
        mm = _VER_RE.search(m.group(1))
        if mm:
            want = mm.group(1)
    if not want:
        return "0.4.24"
    major = ".".join(want.split(".")[:2]) + "."
    same_line = [a for a in avail if a.startswith(major)]
    if same_line:
        ge = [a for a in same_line if Version(a) >= Version(want)]
        return min(ge, key=Version) if ge else max(same_line, key=Version)
    ge_any = [a for a in avail if Version(a) >= Version(want)]
    return min(ge_any, key=Version) if ge_any else avail[-1]


def _compile_to_opcodes(source: str) -> Tuple[List[str], str]:
    try:
        import solcx
        from pyevmasm import disassemble_all

        solcx.set_solc_version(_pick_solc(source), silent=True)
        compiled = solcx.compile_source(source, output_values=["bin-runtime"])

        ops: List[str] = []
        for _, data in compiled.items():
            bc = (data.get("bin-runtime") or "").strip()
            if bc.startswith("0x"):
                bc = bc[2:]
            if len(bc) < 4:
                continue
            if len(bc) % 2:
                bc = bc[:-1]
            try:
                raw = bytes.fromhex(bc)
            except ValueError:
                continue
            ops.extend(instr.name for instr in disassemble_all(raw))
        if ops:
            return ops, "compiled"
    except Exception:
        pass
    return _source_to_opcode_proxy(source), "fallback"


# ── Public API ─────────────────────────────────────────────────────────────
def build_opcode_sequences(texts: List[str]) -> Tuple[List[List[str]], List[str], dict]:
    """Returns (opcode_sequences, per_contract_status, stats)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, dict] = {}
    if OPCODE_CACHE.exists():
        with OPCODE_CACHE.open(encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  [ETH2Vec] Loaded {len(cache)} cached opcode sequences.")

    seqs: List[List[str]] = []
    statuses: List[str] = []
    stats = {"compiled": 0, "fallback": 0, "cached": 0}

    for text in tqdm(texts, desc="  [ETH2Vec] compile/disassemble"):
        key = hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()
        if key in cache:
            stats["cached"] += 1
        else:
            ops, status = _compile_to_opcodes(text)
            cache[key] = {"opcodes": ops, "status": status}
        entry = cache[key]
        seqs.append(entry["opcodes"])
        statuses.append(entry["status"])

    # recompute compiled/fallback from final statuses (covers cache hits)
    stats["compiled"] = sum(1 for s in statuses if s == "compiled")
    stats["fallback"] = sum(1 for s in statuses if s == "fallback")

    with OPCODE_CACHE.open("w", encoding="utf-8") as f:
        json.dump(cache, f)

    print(f"  [ETH2Vec] compiled={stats['compiled']}, fallback={stats['fallback']}, "
          f"total={len(texts)}")
    return seqs, statuses, stats
