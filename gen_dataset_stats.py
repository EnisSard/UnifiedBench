"""
gen_dataset_stats.py
--------------------
Authoritative dataset + fold statistics for the UnifiedBench build in data/.
Everything is recomputed from train.data_loader.load_records() and the two
manifests, so it cannot drift from what the models actually trained on.

Outputs -> results/dataset_stats.json  and  results/dataset_stats.txt
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from train.config import LABELS, MANIFEST, SYNTHETIC_MANIFEST, N_REPEATS, REPEAT_SEED_BASE, RANDOM_SEED
from train.data_loader import load_records, subsource_of
from train.evaluate import get_cv_splits

OUT_JSON = Path("results/dataset_stats.json")
OUT_TXT = Path("results/dataset_stats.txt")

texts, labels, groups, sources, filenames = load_records()
N = len(texts)
lab_arr = np.array(labels)
sub = np.array([subsource_of(f) for f in filenames], dtype=object)
grp = np.array(groups, dtype=object)
SUBS = ["solidifi", "smartbugs", "cgt"]

# ---- per-class x sub-source -------------------------------------------------
per_class = {LABELS[c]: {s: int(np.sum((lab_arr == c) & (sub == s))) for s in SUBS} for c in range(len(LABELS))}
for c in per_class:
    per_class[c]["realworld"] = per_class[c]["smartbugs"] + per_class[c]["cgt"]
    per_class[c]["total"] = per_class[c]["solidifi"] + per_class[c]["realworld"]

# ---- real-world candidates and exclusions ------------------------------------
rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
inc = [r for r in rows if r["inclusion_decision"] == "Included"]
exc = [r for r in rows if r["inclusion_decision"] != "Included"]
excl_reasons = Counter(r["inclusion_decision"] for r in exc)
cgt_origin = Counter(r["original_dataset"] for r in inc if r["original_dataset"].startswith("CGT"))
multi_sets = Counter(r["label_set"] for r in exc if "multi_label" in r["inclusion_decision"])

# ---- SolidiFI base contracts -------------------------------------------------
syn = list(csv.DictReader(SYNTHETIC_MANIFEST.open(encoding="utf-8")))
n_bases = len({r["base_contract"] for r in syn})

# ---- groups -------------------------------------------------------------------
gsize = Counter(groups)
group_stats = {
    "total": len(gsize),
    "solidifi": len(set(grp[sub == "solidifi"])),
    "realworld": len(set(grp[sub != "solidifi"])),
    "singletons": sum(v == 1 for v in gsize.values()),
    "largest": sorted(gsize.values())[-5:],
    "solidifi_base_contracts": n_bases,
}

# ---- fold distribution (seed RANDOM_SEED) and fold sizes over all repeat seeds ----
def fold_table(seed):
    out = []
    for k, (tr, te) in enumerate(get_cv_splits(labels, groups, seed=seed), 1):
        fc = {}
        for ci, cl in enumerate(LABELS):
            m = te[lab_arr[te] == ci]
            fc[cl] = {"total": int(len(m)), "solidifi": int(np.sum(sub[m] == "solidifi")),
                      "realworld": int(np.sum(sub[m] != "solidifi"))}
        out.append({"fold": k, "n_train": int(len(tr)), "n_test": int(len(te)),
                    "test_solidifi": int(np.sum(sub[te] == "solidifi")),
                    "test_realworld": int(np.sum(sub[te] != "solidifi")), "per_class": fc})
    return out

folds = fold_table(RANDOM_SEED)
repeat_seeds = [REPEAT_SEED_BASE + r for r in range(N_REPEATS)]
fold_sizes = {s: [len(te) for _, te in get_cv_splits(labels, groups, seed=s)] for s in repeat_seeds}

data = {
    "total_contracts": N,
    "source_totals": {s: int(np.sum(sub == s)) for s in SUBS},
    "per_class_per_source": per_class,
    "groups": group_stats,
    "realworld_candidates": len(rows),
    "realworld_included": len(inc),
    "realworld_excluded": len(exc),
    "exclusion_reasons": dict(excl_reasons),
    "excluded_multilabel_label_sets": dict(multi_sets),
    "cgt_included_by_origin": dict(cgt_origin),
    "folds_seed_%d" % RANDOM_SEED: folds,
    "test_fold_sizes_by_repeat_seed": fold_sizes,
}
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")

# ---- text report ------------------------------------------------------------
L = []
L.append("=" * 74)
L.append(f"  UNIFIEDBENCH -- {N} contracts")
L.append("=" * 74)
L.append(f"  {'Class':<28}{'SolidiFI':>9}{'SmartBugs':>10}{'CGT':>6}{'RealWorld':>10}{'Total':>7}")
L.append("  " + "-" * 70)
for cl in LABELS:
    v = per_class[cl]
    L.append(f"  {cl:<28}{v['solidifi']:>9}{v['smartbugs']:>10}{v['cgt']:>6}{v['realworld']:>10}{v['total']:>7}")
L.append("  " + "-" * 70)
t = data["source_totals"]
L.append(f"  {'TOTAL':<28}{t['solidifi']:>9}{t['smartbugs']:>10}{t['cgt']:>6}"
         f"{t['smartbugs'] + t['cgt']:>10}{N:>7}")
L.append("")
L.append(f"  SolidiFI: {len(syn)} files from {n_bases} base contracts")
L.append(f"  Clone-family groups: {group_stats['total']} (SolidiFI {group_stats['solidifi']}, "
         f"real-world {group_stats['realworld']}); singletons {group_stats['singletons']}; "
         f"largest {group_stats['largest']}")
L.append(f"  Real-world candidates: {len(rows)} -> included {len(inc)}, excluded {len(exc)}")
L.append(f"  Exclusion reasons: {dict(excl_reasons)}")
L.append(f"  Excluded multi-label CGT label sets: {dict(multi_sets.most_common())}")
L.append(f"  Included CGT contracts by originating campaign(s): {dict(cgt_origin.most_common())}")
L.append("")
L.append(f"  FOLD TEST-SET DISTRIBUTION, seed {RANDOM_SEED}  (total (solidifi/realworld))")
L.append("  " + "-" * 70)
L.append(f"  {'Class':<26}" + "".join(f"{'F' + str(k):>9}" for k in range(1, 6)))
for cl in LABELS:
    row = f"  {cl:<26}"
    for f in folds:
        c = f["per_class"][cl]
        row += f"{str(c['total']) + '(' + str(c['solidifi']) + '/' + str(c['realworld']) + ')':>9}"
    L.append(row)
L.append("  " + "-" * 70)
L.append(f"  {'n_test':<26}" + "".join(f"{f['n_test']:>9}" for f in folds))
L.append(f"  {'test solidifi':<26}" + "".join(f"{f['test_solidifi']:>9}" for f in folds))
L.append(f"  {'test realworld':<26}" + "".join(f"{f['test_realworld']:>9}" for f in folds))
L.append("")
L.append("  Test-fold sizes for every repeat seed:")
for s, sz in fold_sizes.items():
    L.append(f"    seed {s}: {sz}")
report = "\n".join(L)
OUT_TXT.write_text(report, encoding="utf-8")
print(report)
print(f"\n[OK] wrote {OUT_JSON} and {OUT_TXT}")
