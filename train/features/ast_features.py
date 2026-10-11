"""
train/features/ast_features.py
-------------------------------
SmartEmbed-inspired feature extraction (after Gao et al., ICSME 2019).

Pipeline:
  1. Normalize Solidity source: replace variable names → VAR, literals → LIT,
     addresses → ADDR, keeping Solidity keywords and structural tokens intact.
  2. Tokenize the normalized stream.
  3. Train Word2Vec Skip-Gram on the token streams of the training split.
  4. Represent each contract as the mean-pool of its token embeddings (100-dim).
"""

import re
from typing import List
import numpy as np
from gensim.models import Word2Vec
from train.config import SMARTEMBED_PARAMS, get_seed

# ── Solidity keywords to KEEP as-is ──────────────────────────────────────────
_KEYWORDS = {
    "pragma", "solidity", "contract", "library", "interface", "is",
    "function", "modifier", "event", "struct", "enum", "mapping",
    "address", "uint", "int", "bool", "bytes", "string", "memory",
    "storage", "calldata", "public", "private", "internal", "external",
    "pure", "view", "payable", "returns", "return", "if", "else",
    "for", "while", "do", "break", "continue", "throw", "require",
    "assert", "revert", "emit", "new", "delete", "this", "super",
    "msg", "tx", "block", "now", "transfer", "send", "call", "delegatecall",
    "selfdestruct", "suicide", "keccak256", "sha256", "ecrecover",
    "true", "false",
}

# Regex: strip comments first
_STRIP_COMMENTS = re.compile(r"//.*?$|/\*.*?\*/", re.DOTALL | re.MULTILINE)
# Hex/decimal/string literals
_HEX_LIT  = re.compile(r"0x[0-9a-fA-F]+")
_STR_LIT  = re.compile(r'"[^"]*"|\'[^\']*\'')
_NUM_LIT  = re.compile(r"\b\d+\b")
# Address pattern (0x + 40 hex chars)
_ADDR_LIT = re.compile(r"0x[0-9a-fA-F]{40}")


def _normalize(source: str) -> List[str]:
    """Normalize a Solidity source file to a list of structural tokens."""
    text = _STRIP_COMMENTS.sub(" ", source)
    text = _ADDR_LIT.sub(" ADDR ", text)
    text = _STR_LIT.sub(" LIT ", text)
    text = _HEX_LIT.sub(" LIT ", text)
    text = _NUM_LIT.sub(" LIT ", text)

    # Tokenize: split on whitespace + punctuation boundaries
    raw_tokens = re.split(r"[\s;,{}\[\]()\.<>=!+\-*/&|^~?:]+", text)

    tokens: List[str] = []
    for tok in raw_tokens:
        tok = tok.strip()
        if not tok:
            continue
        # Keep keywords as-is; replace identifiers (camelCase, snake_case) with VAR
        if tok.lower() in _KEYWORDS or tok in _KEYWORDS:
            tokens.append(tok.lower())
        elif re.match(r"^[a-zA-Z_]\w*$", tok):
            tokens.append("VAR")
        elif tok == "LIT" or tok == "ADDR":
            tokens.append(tok)
        # skip pure punctuation leftovers

    return tokens if tokens else ["EMPTY"]


# ── Word2Vec training ─────────────────────────────────────────────────────────

def build_word2vec(token_streams: List[List[str]]) -> Word2Vec:
    """Train Word2Vec Skip-Gram on the given token streams (the training split of a fold)."""
    params = dict(SMARTEMBED_PARAMS, seed=get_seed())   # seed of the current run
    model = Word2Vec(
        sentences=token_streams,
        **params,
    )
    return model


def embed(token_stream: List[str], w2v: Word2Vec) -> np.ndarray:
    """Mean-pool Word2Vec token vectors → 100-dim contract embedding."""
    vecs = [
        w2v.wv[tok]
        for tok in token_stream
        if tok in w2v.wv
    ]
    if not vecs:
        return np.zeros(SMARTEMBED_PARAMS["vector_size"], dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)
