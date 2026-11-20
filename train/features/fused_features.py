import re
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

from train.config import get_seed

STATIC_FEATURE_NAMES = [
    "external_calls", "transfer_calls", "send_calls", "delegatecall",
    "selfdestruct", "suicide", "unchecked_math", "SafeMath_usage",
    "require_statements", "modifier_count", "onlyOwner", "mutex_locks",
    "blockhash_usage", "block_timestamp", "now_usage", "public_functions",
    "private_functions", "external_functions", "event_count", "lines_of_code",
    "contract_count", "function_count"
]

def extract_static_features(source_code: str) -> np.ndarray:
    """Extracts 22 hand-crafted static features via regex."""
    features = []
    
    features.append(len(re.findall(r'\.call\(', source_code)))
    features.append(len(re.findall(r'\.transfer\(', source_code)))
    features.append(len(re.findall(r'\.send\(', source_code)))
    features.append(len(re.findall(r'delegatecall\(', source_code)))
    features.append(len(re.findall(r'selfdestruct\(', source_code)))
    features.append(len(re.findall(r'suicide\(', source_code)))
    features.append(len(re.findall(r'(\+=|-=|\*=|/=|%=)', source_code)))
    features.append(1.0 if 'SafeMath' in source_code else 0.0)
    features.append(len(re.findall(r'require\(', source_code)))
    features.append(len(re.findall(r'\bmodifier\s+\w+', source_code)))
    features.append(1.0 if 'onlyOwner' in source_code else 0.0)
    features.append(1.0 if re.search(r'\b(lock|mutex|unlocked|locked)\b', source_code, re.IGNORECASE) else 0.0)
    features.append(len(re.findall(r'blockhash\(', source_code)))
    features.append(len(re.findall(r'block\.timestamp', source_code)))
    features.append(len(re.findall(r'\bnow\b', source_code)))
    features.append(len(re.findall(r'\bfunction\s+[^{]*?\bpublic\b', source_code)))
    features.append(len(re.findall(r'\bfunction\s+[^{]*?\bprivate\b', source_code)))
    features.append(len(re.findall(r'\bfunction\s+[^{]*?\bexternal\b', source_code)))
    features.append(len(re.findall(r'\bevent\s+\w+', source_code)))
    features.append(float(source_code.count('\n') + 1))
    features.append(len(re.findall(r'\bcontract\s+\w+', source_code)))
    features.append(len(re.findall(r'\bfunction\s+\w+', source_code)))
    
    vec = np.array(features, dtype=np.float32)
    max_val = np.max(np.abs(vec))
    if max_val > 0:
        vec = vec / max_val
        
    return vec

class FusedFeatureExtractor:
    def __init__(self, device='cpu'):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
    def _chunked_embeddings(self, texts: List[str]) -> np.ndarray:
        """Mean of the MiniLM embeddings of up to MINILM_MAX_CHUNKS evenly spaced
        250-token chunks, so the vector reflects the whole contract and not only
        its first 256 tokens."""
        from train.config import MINILM_MAX_CHUNKS
        tok = self.embedder.tokenizer
        chunk_texts, owner = [], []
        for i, t in enumerate(texts):
            ids = tok(t, add_special_tokens=False)["input_ids"]
            pieces = [ids[j:j + 250] for j in range(0, max(len(ids), 1), 250)] or [[]]
            pieces = select_chunks(pieces, MINILM_MAX_CHUNKS, tok.decode)
            for p in pieces:
                chunk_texts.append(tok.decode(p) if p else "")
                owner.append(i)
        emb = self.embedder.encode(chunk_texts, batch_size=64, show_progress_bar=True,
                                   convert_to_numpy=True)
        owner = np.asarray(owner)
        return np.vstack([emb[owner == i].mean(axis=0) for i in range(len(texts))])

    def transform(self, texts: List[str]) -> np.ndarray:
        from train.config import PROFILE
        print("    [Fused] Extracting semantic embeddings...")
        if PROFILE["chunking"]:
            semantic_embeddings = self._chunked_embeddings(texts)
        else:
            truncated_texts = [t[:10000] for t in texts]
            semantic_embeddings = self.embedder.encode(
                truncated_texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            ) # 384-d

        print("    [Fused] Extracting static features...")
        static_features = np.array([extract_static_features(t) for t in texts]) # 22-d

        fused = np.concatenate([semantic_embeddings, static_features], axis=1)
        return fused


def pick_evenly(items: list, k: int) -> list:
    """k items spread evenly over `items` (always including the first and last)."""
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]


def select_chunks(pieces: list, k: int, decode) -> list:
    """Chunk choice for long contracts: evenly spaced (runA) or risk-ranked (runB)."""
    from train.config import PROFILE
    if len(pieces) <= k or PROFILE.get("chunk_select", "even") == "even":
        return pick_evenly(pieces, k)
    from train.text_utils import pick_by_risk
    return pick_by_risk(pieces, k, [decode(p) for p in pieces])


_FUSED_CACHE: dict = {}


def fused_features(texts: List[str], device: str = 'cpu') -> np.ndarray:
    """406-d fused vectors for `texts`. MiniLM is a frozen pre-trained encoder and the
    static features are fixed regexes (nothing is fitted on labels), so the result is
    cached in memory and on disk and reused by every fused model, repeat and restart."""
    import hashlib
    from train.config import CACHE_DIR, PROFILE
    key = hashlib.sha256(("\x00".join(texts) + PROFILE["name"] + str(PROFILE["chunking"])
                         + ("" if PROFILE.get("chunk_select", "even") == "even" else PROFILE["chunk_select"]))
                         .encode("utf-8", "ignore")).hexdigest()
    if key not in _FUSED_CACHE:
        f = CACHE_DIR / f"fused_{key[:16]}.npy"
        if f.exists():
            _FUSED_CACHE[key] = np.load(f)
        else:
            X = FusedFeatureExtractor(device=device).transform(texts)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(f, X)
            _FUSED_CACHE[key] = X
    return _FUSED_CACHE[key]

def apply_gaussian_augmentation(
    X_train: np.ndarray, y_train, num_classes: int, seed: int | None = None, weights=None
):
    """Gaussian-noise augmentation of the training split (used by the fused-feature models).

    Every training instance of classes 0 .. num_classes-2 (all classes except the
    last one, unchecked_low_level_calls) is replicated three times with additive
    N(0, 0.01^2) noise; the last class is left untouched. Class-weighted loss
    remains the primary mechanism for class imbalance. Only the training split
    is ever augmented. The noise comes from a generator seeded with `seed`, so
    the augmentation is reproducible.
    """
    y_train = np.array(y_train)
    rng = np.random.default_rng(get_seed() if seed is None else seed)
    print(f"    [Augment] Applying Gaussian noise to classes 0-{num_classes-2} (x3)...")
    aug_X = [X_train]
    aug_y = [y_train]
    w_train = None if weights is None else np.asarray(weights, dtype=float)
    aug_w = [w_train]

    for c in range(num_classes - 1):
        idx = np.where(y_train == c)[0]

        if len(idx) == 0: continue

        X_c = X_train[idx]
        y_c = y_train[idx]

        for _ in range(3):
            noise = rng.normal(loc=0.0, scale=0.01, size=X_c.shape).astype(np.float32)
            aug_X.append(X_c + noise)
            aug_y.append(y_c)
            if w_train is not None:
                aug_w.append(w_train[idx])

    X_out, y_out = np.concatenate(aug_X, axis=0), np.concatenate(aug_y, axis=0)
    if w_train is None:
        return X_out, y_out
    return X_out, y_out, np.concatenate(aug_w, axis=0)
