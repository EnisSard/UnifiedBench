"""train/data_loader.py — Load the two manifests + raw .sol source texts."""
import csv
from typing import Tuple, List

from train.config import EXPANDED_DIR, SYNTHETIC_DIR, MANIFEST, SYNTHETIC_MANIFEST, LABEL2IDX, PROFILE, set_rows
from train.text_utils import strip_comments, neutralize_solidifi_ids, assert_no_annotations


def subsource_of(filename: str) -> str:
    """'solidifi' | 'smartbugs' | 'cgt' from a unified contract identifier."""
    if filename.startswith("solidifi__"):
        return "solidifi"
    if filename.startswith("sb_curated__"):
        return "smartbugs"
    if filename.startswith("cgt__"):
        return "cgt"
    raise ValueError(f"unknown contract origin: {filename}")


def load_records(strip_comments_from_inputs: bool = True) -> Tuple[List[str], List[int], List[str], List[str], List[str]]:
    """
    Returns (texts, labels, group_ids, source_datasets, filenames) in a fixed order
    (SolidiFI manifest order, then real-world manifest order).

    group_ids                   clone-family id from build_unified_dataset.py; SolidiFI files derived
                                from the same base contract share one id
    source_datasets             'solidifi' | 'realworld' (use subsource_of(filename) to split the
                                real-world set into SmartBugs Curated and CGT)
    strip_comments_from_inputs  if True (default), strips comments from raw text to eliminate
                                ground-truth label leakage from SmartBugs curator tags.
    """
    texts, labels, group_ids, source_datasets, filenames = [], [], [], [], []
    label_sets = []

    def _add(path, label_idx, group, source, name, label_set=None):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if strip_comments_from_inputs:
            text = strip_comments(text)
        if source == "solidifi" and PROFILE["neutralize_solidifi"]:
            text = neutralize_solidifi_ids(text)
        texts.append(text)
        labels.append(label_idx)
        group_ids.append(group)
        source_datasets.append(source)
        filenames.append(name)
        label_sets.append(frozenset(label_set or [label_idx]))

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
            path = EXPANDED_DIR / row["category"] / row["contract_identifier"]
            lset = sorted(LABEL2IDX[c] for c in row["label_set"].split("+"))
            if PROFILE["multilabel"] and len(lset) > 1:
                for lab in lset:        # one row per label, same group and text
                    _add(path, lab, row["group_identifier"], "realworld",
                         row["contract_identifier"], lset)
            else:
                _add(path, LABEL2IDX[row["category"]], row["group_identifier"], "realworld",
                     row["contract_identifier"])

    if strip_comments_from_inputs:
        assert_no_annotations(texts, filenames)

    set_rows(filenames, label_sets)
    if PROFILE["multilabel"]:
        n_ml = len({n for n, ls in zip(filenames, label_sets) if len(ls) > 1})
        print(f"[DATA] multi-label: {n_ml} contracts expanded to "
              f"{sum(len(ls) > 1 for ls in label_sets)} rows (one per label, weight 1/|set|)")
    n_sb = sum(subsource_of(n) == "smartbugs" for n in filenames)
    n_cgt = sum(subsource_of(n) == "cgt" for n in filenames)
    print(f"[DATA] Loaded {len(texts)} contracts "
          f"({source_datasets.count('solidifi')} SolidiFI, "
          f"{source_datasets.count('realworld')} RealWorld = {n_sb} SmartBugs + {n_cgt} CGT; "
          f"{len(set(group_ids))} clone-family groups; profile={PROFILE['name']})")
    return texts, labels, group_ids, source_datasets, filenames
