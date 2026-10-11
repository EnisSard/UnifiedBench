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
        
    def transform(self, texts: List[str]) -> np.ndarray:
        print("    [Fused] Extracting semantic embeddings...")
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


_FUSED_CACHE: dict = {}


def fused_features(texts: List[str], device: str = 'cpu') -> np.ndarray:
    """406-d fused vectors for `texts`. MiniLM is a frozen pre-trained encoder and the
    static features are fixed regexes (nothing is fitted on labels), so the result is
    cached per process and reused by every fused model and every repeat."""
    import hashlib
    key = hashlib.sha256("\x00".join(texts).encode("utf-8", "ignore")).hexdigest()
    if key not in _FUSED_CACHE:
        _FUSED_CACHE[key] = FusedFeatureExtractor(device=device).transform(texts)
    return _FUSED_CACHE[key]

def apply_gaussian_augmentation(
    X_train: np.ndarray, y_train, num_classes: int, seed: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
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

    for c in range(num_classes - 1):
        idx = np.where(y_train == c)[0]

        if len(idx) == 0: continue

        X_c = X_train[idx]
        y_c = y_train[idx]

        for _ in range(3):
            noise = rng.normal(loc=0.0, scale=0.01, size=X_c.shape).astype(np.float32)
            aug_X.append(X_c + noise)
            aug_y.append(y_c)
            
    return np.concatenate(aug_X, axis=0), np.concatenate(aug_y, axis=0)
