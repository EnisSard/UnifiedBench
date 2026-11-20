"""
train/text_utils.py
-------------------
Source-text preprocessing shared by the dataset builder and the data loader.

strip_comments           removes // and /* */ comments with a small scanner that
                         understands string literals, so "https://..." inside a
                         string is kept intact. Block comments are replaced by the
                         newlines they contained, so line structure is preserved.
neutralize_solidifi_ids  renames SolidiFI's injected identifiers, which spell out
                         the weakness type (e.g. balances_re_ent, bug_tmstmp1), to
                         neutral per-file names (sym1, sym2, ...).
assert_no_annotations    fails loudly if a curator annotation survived into a
                         model input.
"""
import re
from typing import Iterable

# ── Comment stripping ─────────────────────────────────────────────────────────

def strip_comments(code: str) -> str:
    out = []
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if c in ('"', "'"):                       # string literal: copy verbatim
            q = c
            j = i + 1
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == q or code[j] == "\n":   # unterminated strings end at EOL
                    j += 1
                    break
                j += 1
            out.append(code[i:j])
            i = j
        elif c == "/" and nxt == "/":             # line comment
            j = code.find("\n", i)
            i = n if j == -1 else j               # keep the newline itself
        elif c == "/" and nxt == "*":             # block comment
            j = code.find("*/", i + 2)
            end = n if j == -1 else j + 2
            out.append("\n" * code.count("\n", i, end))
            i = end
        else:
            out.append(c)
            i += 1
    return "".join(out)


# ── SolidiFI identifier neutralisation ────────────────────────────────────────

# Identifiers SolidiFI gives its injected code: they name the weakness type.
_SOLIDIFI_MARKERS = re.compile(
    r"\b(?:[A-Za-z0-9_]*(?:_re_ent|_tmstmp|_TOD|_intou|_unchk|_txorigin)[A-Za-z0-9_]*"
    r"|bugv?_[A-Za-z0-9_]+)\b"
)


def neutralize_solidifi_ids(code: str) -> str:
    """Rename every weakness-named identifier to symN, numbered by first appearance
    in this file, so names carry no class information (the same name maps to the
    same symbol within a file, keeping the code's references consistent)."""
    mapping = {}
    existing = set(re.findall(r"\bsym\d+\b", code))

    def repl(m):
        name = m.group(0)
        if name not in mapping:
            k = len(mapping) + 1
            while f"sym{k}" in existing:
                k += 1
            mapping[name] = f"sym{k}"
            existing.add(mapping[name])
        return mapping[name]

    return _SOLIDIFI_MARKERS.sub(repl, code)


def has_solidifi_markers(code: str) -> bool:
    return _SOLIDIFI_MARKERS.search(code) is not None


# ── Leak guard ────────────────────────────────────────────────────────────────

_ANNOTATION = re.compile(r"<yes>\s*<report>|@vulnerable_at_lines|@source:|@author:")


def assert_no_annotations(texts: Iterable[str], names: Iterable[str]) -> None:
    bad = [n for t, n in zip(texts, names) if _ANNOTATION.search(t)]
    if bad:
        raise RuntimeError(
            f"Curator annotations found in {len(bad)} model inputs (label leak), e.g. {bad[:3]}. "
            "Rebuild the dataset without --keep-comments or load with comment stripping on."
        )


# ── Risk-ranked chunk selection (profile runB) ───────────────────────────────
# Security-relevant operations, a few per class, used only to decide which chunks
# of a long contract the model reads. The list is fixed a priori (never fitted on
# labels) and covers all six classes so no class is favoured.
_RISK = [re.compile(p) for p in (
    r"tx\.origin|selfdestruct|suicide|delegatecall|callcode|onlyOwner|\bowner\b|msg\.sender\s*==",  # access control
    r"\+=|-=|\*=|\+\+|\bSafeMath\b|\b(?:add|sub|mul)\s*\(",                                    # arithmetic
    r"\bapprove\b|transferFrom|keccak256|sha3|\breward\b|\bsolution\b|\bprice\b",              # front-running
    r"\.call\.value|\.call\{|\bbalances?\s*\[|\.transfer\(",                                    # reentrancy
    r"block\.timestamp|\bnow\b|block\.number|blockhash",                                           # time
    r"\.call\(|\.send\(|\.delegatecall\(|\.callcode\(",                                          # low-level calls
)]


def risk_score(text: str) -> int:
    """Per-category hit counts capped at 3 and summed, so one very frequent
    operation (e.g. ++ in loops) cannot dominate the choice."""
    return sum(min(len(r.findall(text)), 3) for r in _RISK)


def pick_by_risk(pieces: list, k: int, piece_texts: list) -> list:
    """Keep the first piece (pragma, state variables) plus the k-1 pieces with the
    most security-relevant operations, in their original order."""
    if len(pieces) <= k:
        return pieces
    scores = [risk_score(t) for t in piece_texts]
    rest = sorted(range(1, len(pieces)), key=lambda i: (-scores[i], i))[: k - 1]
    return [pieces[i] for i in sorted([0] + rest)]
