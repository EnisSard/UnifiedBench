"""train/data_loader.py — Load the two manifests + raw .sol source texts."""
import csv
from typing import Tuple, List

from train.config import EXPANDED_DIR, SYNTHETIC_DIR, MANIFEST, SYNTHETIC_MANIFEST, LABEL2IDX


def subsource_of(filename: str) -> str:
    """'solidifi' | 'smartbugs' | 'cgt' from a unified contract identifier."""
    if filename.startswith("solidifi__"):
        return "solidifi"
    if filename.startswith("sb_curated__"):
        return "smartbugs"
    if filename.startswith("cgt__"):
        return "cgt"
    raise ValueError(f"unknown contract origin: {filename}")


def load_records() -> Tuple[List[str], List[int], List[str], List[str], List[str]]:
    """
    Returns (texts, labels, group_ids, source_datasets, filenames) in a fixed order
    (SolidiFI manifest order, then real-world manifest order).

    group_ids       clone-family id from build_unified_dataset.py; SolidiFI files derived
                    from the same base contract share one id
    source_datasets 'solidifi' | 'realworld'  (use subsource_of(filename) to split the
                    real-world set into SmartBugs Curated and CGT)
    """
    texts, labels, group_ids, source_datasets, filenames = [], [], [], [], []

    def _add(path, label_idx, group, source, name):
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
        labels.append(label_idx)
        group_ids.append(group)
        source_datasets.append(source)
        filenames.append(name)

    # 1. Synthetic benchmark (SolidiFI)
    with SYNTHETIC_MANIFEST.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            _add(SYNTHETIC_DIR / row["category"] / row["contract_identifier"],
                 LABEL2IDX[row["category"]], row["group_identifier"], "solidifi",
                 row["contract_identifier"])

    # 2. Expanded real-world set (SmartBugs Curated + CGT)
    with MANIFEST.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["inclusion_decision"] != "Included":
                continue
            _add(EXPANDED_DIR / row["category"] / row["contract_identifier"],
                 LABEL2IDX[row["category"]], row["group_identifier"], "realworld",
                 row["contract_identifier"])

    n_sb = sum(subsource_of(n) == "smartbugs" for n in filenames)
    n_cgt = sum(subsource_of(n) == "cgt" for n in filenames)
    print(f"[DATA] Loaded {len(texts)} contracts "
          f"({source_datasets.count('solidifi')} SolidiFI, "
          f"{source_datasets.count('realworld')} RealWorld = {n_sb} SmartBugs + {n_cgt} CGT; "
          f"{len(set(group_ids))} clone-family groups)")
    return texts, labels, group_ids, source_datasets, filenames
