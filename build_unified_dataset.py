"""
build_unified_dataset.py
------------------------
Builds the UnifiedBench dataset from the source corpora cloned by setup_data.py:

  data/synthetic_benchmark/<class>/        SolidiFI injected contracts (synthetic)
  data/synthetic_benchmark/synthetic_manifest.csv
                                           base contract, normalised hash and
                                           clone-family group id of every SolidiFI file
  data/expanded_real_world_set/<class>/    SmartBugs Curated + the selected
                                           Consolidated Ground Truth (CGT) contracts
  data/expanded_real_world_set/reproducibility_manifest.csv
                                           provenance, CGT label set, normalised SHA-256
                                           hash, clone-family group id and inclusion
                                           decision of every real-world candidate

Selection rules
  * SolidiFI: every injected contract of the seven weakness folders that map onto
    the six canonical classes.
  * SmartBugs Curated: every file of the six canonical DASP folders (125 files).
    One contract (0x627fa62...) is filed under both reentrancy and
    unchecked_low_level_calls; both files are kept, each with its folder label, and
    their identical sources put them in the same clone family.
  * CGT: every entry whose property holds, whose SWC code maps onto a canonical class
    (entries originating from SolidiFI are skipped) and whose PRAGMA-INSENSITIVE
    fingerprint fp_sol2 names a stored source file, source/<fp_sol2>.sol. CGT stores
    sources under the full fingerprint fp_sol, and fp_sol2 equals fp_sol when the
    source declares no compiler-version pragma, so this rule selects, in practice,
    contracts without a version pragma (all 120 retained source files). CGT lists one row
    per (contract, property), so a contract can carry several canonical classes. All
    rows of a contract -- and of every contract with the same normalised source --
    are merged into one label set, and --cgt-multilabel decides what happens to
    contracts whose label set has more than one class:
        drop    (default) exclude them; only unambiguous single-label contracts remain
        rarest  keep them under the class that is rarest among the CGT candidates
  * Exact duplicates (same comment/whitespace-stripped source): SmartBugs Curated
    contracts are always kept; a CGT contract identical to an already-kept file is
    excluded.

Clone-family groups (the key StratifiedGroupKFold uses, so a family never spans
a fold boundary) are built with Union-Find over the WHOLE corpus:
  1. every SolidiFI file derived from the same base contract (buggy_<N>.sol in
     each weakness folder) shares a group;
  2. files with identical normalised source share a group;
  3. any two contracts -- across SolidiFI and real-world alike -- whose word-level
     token sets have Jaccard similarity >= 0.80 share a group.
"""
import argparse
import csv
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.resolve()
DATA_DIR       = PROJECT_ROOT / "data"
SOLIDIFI_ROOT  = DATA_DIR / "SolidiFI-benchmark"
SMARTBUGS_ROOT = DATA_DIR / "smartbugs-curated"
CGT_ROOT       = DATA_DIR / "CGT"

SYNTHETIC_DIR  = DATA_DIR / "synthetic_benchmark"
EXPANDED_DIR   = DATA_DIR / "expanded_real_world_set"

JACCARD_THRESHOLD = 0.80

# ──────────────────────────────────────────────
# Categories and Mapping
# ──────────────────────────────────────────────
CANONICAL_LABELS = [
    "access_control",
    "arithmetic",
    "front_running",
    "reentrancy",
    "time_manipulation",
    "unchecked_low_level_calls",
]

# Mapping SWC directly to target categories
SWC_MAP = {
    101: "arithmetic",
    104: "unchecked_low_level_calls",
    107: "reentrancy",
    114: "front_running",
    116: "time_manipulation",
    # Access control map
    105: "access_control",
    106: "access_control",
    108: "access_control",
    112: "access_control",
    115: "access_control",
}

SMARTBUGS_MAP = {c: c for c in CANONICAL_LABELS}

SOLIDIFI_MAP = {
    "Re-entrancy": "reentrancy",
    "Overflow-Underflow": "arithmetic",
    "Timestamp-Dependency": "time_manipulation",
    "TOD": "front_running",
    "Unchecked-Send": "unchecked_low_level_calls",
    "Unhandled-Exceptions": "unchecked_low_level_calls",
    "tx.origin": "access_control",
}

# ──────────────────────────────────────────────
# Normalisation, hashing, similarity
# ──────────────────────────────────────────────

def strip_comments(code: str) -> str:
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return re.sub(r'//.*', '', code)


def normalize_solidity(code: str) -> str:
    """Strip comments and all whitespace (used for exact-duplicate hashing)."""
    return re.sub(r'\s+', '', strip_comments(code))


def norm_hash(code: str) -> str:
    return hashlib.sha256(normalize_solidity(code).encode('utf-8')).hexdigest()


def tokenize(code: str) -> set:
    """Word-level token set (identifiers, keywords, numbers) of the comment-stripped source."""
    return set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+', strip_comments(code)))


def jaccard_sim(set1: set, set2: set) -> float:
    if not set1 and not set2: return 1.0
    return len(set1 & set2) / len(set1 | set2)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)

# ──────────────────────────────────────────────
# Collection
# ──────────────────────────────────────────────

def safe_stem(path: Path, prefix: str, seen: dict) -> str:
    base = f"{prefix}__{path.name}"
    if base not in seen:
        seen[base] = 0
        return base
    seen[base] += 1
    return f"{prefix}__{path.stem}_{seen[base]}{path.suffix}"


def collect_solidifi() -> list[dict]:
    records = []
    base = SOLIDIFI_ROOT / "buggy_contracts"
    if not base.is_dir():
        for alt in SOLIDIFI_ROOT.iterdir():
            if alt.is_dir() and "buggy" in alt.name.lower():
                base = alt; break
    if not base.is_dir():
        raise SystemExit(f"SolidiFI not found under {SOLIDIFI_ROOT} -- run setup_data.py first")

    seen = {}
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or folder.name not in SOLIDIFI_MAP:
            continue
        for sol in sorted(folder.glob("*.sol")):
            m = re.match(r"buggy_(\d+)\.sol$", sol.name)
            if not m:
                raise SystemExit(f"Unexpected SolidiFI file name: {sol}")
            records.append({
                "source_path": sol,
                "category": SOLIDIFI_MAP[folder.name],
                "original_label": folder.name,
                "base_contract": f"buggy_{int(m.group(1))}",
                "dest_name": safe_stem(sol, "solidifi", seen),
            })
    return records


def collect_smartbugs() -> list[dict]:
    records = []
    dataset_dir = SMARTBUGS_ROOT / "dataset"
    if not dataset_dir.is_dir():
        dataset_dir = SMARTBUGS_ROOT

    seen = {}
    for folder in sorted(dataset_dir.iterdir()):
        if not folder.is_dir() or folder.name not in SMARTBUGS_MAP:
            continue
        canonical = SMARTBUGS_MAP[folder.name]
        for sol in sorted(folder.rglob("*.sol")):
            records.append({
                "source_path": sol,
                "category": canonical,
                "label_set": canonical,
                "original_label": folder.name,
                "source_dataset": "smartbugs-curated",
                "dataset_version": "1.0",
                "source_url": "https://github.com/smartbugs/smartbugs-curated",
                "mapping": canonical,
                "dest_name": safe_stem(sol, "sb_curated", seen),
            })
    return records


def collect_cgt() -> list[dict]:
    """One record per CGT source file, carrying ALL canonical classes its rows assert."""
    csv_file = CGT_ROOT / "consolidated.csv"
    if not csv_file.exists():
        raise SystemExit(f"{csv_file} not found -- run setup_data.py first")

    per_file = {}
    with csv_file.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=';'):
            if row.get("property_holds", "").lower() != "t":
                continue                                   # only vulnerable examples
            orig_ds = row.get("dataset", "CGT")
            if orig_ds.lower() == "solidifi":
                continue                                   # SolidiFI is the synthetic corpus
            try:
                swc = int(row.get("swc", ""))
            except ValueError:
                continue
            if swc not in SWC_MAP:
                continue
            fp = row.get("fp_sol2") or row.get("fp_sol")   # pragma-insensitive fingerprint (see docstring)
            if not fp or not (CGT_ROOT / "source" / f"{fp}.sol").exists():
                continue
            rec = per_file.setdefault(fp, {"classes": set(), "datasets": set(),
                                           "swcs": set(), "properties": set()})
            rec["classes"].add(SWC_MAP[swc])
            rec["datasets"].add(orig_ds)
            rec["swcs"].add(swc)
            rec["properties"].add(row.get("property", ""))

    records = []
    seen = {}
    for fp in sorted(per_file):                            # order-independent: every file is considered
        info = per_file[fp]
        src_file = CGT_ROOT / "source" / f"{fp}.sol"
        records.append({
            "source_path": src_file,
            "classes": set(info["classes"]),
            "original_label": " | ".join(sorted(info["properties"])),
            "source_dataset": "CGT (" + "+".join(sorted(info["datasets"])) + ")",
            "dataset_version": "master",
            "source_url": "https://github.com/gsalzer/cgt.git",
            "mapping": "+".join(f"SWC-{s}" for s in sorted(info["swcs"])),
            "dest_name": safe_stem(src_file, "cgt", seen),
        })
    return records

# ──────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────

def build_datasets(cgt_multilabel: str = "drop") -> None:
    synth = collect_solidifi()
    sb = collect_smartbugs()
    cgt = collect_cgt()

    for r in synth + sb + cgt:
        r["code"] = Path(r["source_path"]).read_text(encoding="utf-8", errors="ignore")
        r["norm_hash"] = norm_hash(r["code"])

    # ---- CGT label sets are merged over identical normalised sources ----------
    hash_classes = defaultdict(set)
    for r in cgt:
        hash_classes[r["norm_hash"]] |= r["classes"]
    for r in cgt:
        r["classes"] = set(hash_classes[r["norm_hash"]])
        r["label_set"] = "+".join(sorted(r["classes"]))

    # class frequency among CGT candidates (only used by --cgt-multilabel rarest)
    cgt_freq = Counter(c for r in cgt for c in r["classes"])

    # ---- inclusion decisions ----------------------------------------------------
    kept_hashes = {}
    for r in sb:                                           # SmartBugs Curated is never dropped
        r["included"], r["exclusion_reason"] = True, ""
        kept_hashes.setdefault(r["norm_hash"], r["dest_name"])
    for r in cgt:
        r["included"], r["exclusion_reason"] = False, ""
        if r["norm_hash"] in kept_hashes:
            r["exclusion_reason"] = "exact_duplicate"
            continue
        if len(r["classes"]) > 1:
            if cgt_multilabel == "drop":
                r["exclusion_reason"] = "multi_label"
                continue
            r["category"] = min(r["classes"], key=lambda c: (cgt_freq[c], c))
        else:
            r["category"] = next(iter(r["classes"]))
        r["included"] = True
        kept_hashes[r["norm_hash"]] = r["dest_name"]
    for r in cgt:
        r.setdefault("category", "+".join(sorted(r["classes"])))

    # ---- clone-family groups over the whole corpus ----------------------------
    pool = synth + [r for r in sb + cgt if r["included"]]
    uf = UnionFind(len(pool))
    first_by_key = {}
    for i, r in enumerate(pool):
        keys = [("hash", r["norm_hash"])]
        if "base_contract" in r:
            keys.append(("solidifi_base", r["base_contract"]))
        for key in keys:
            if key in first_by_key:
                uf.union(i, first_by_key[key])
            else:
                first_by_key[key] = i
    tokens = [tokenize(r["code"]) for r in pool]
    print(f"  [CLONES] Jaccard >= {JACCARD_THRESHOLD} over {len(pool)} contracts ...")
    n_jaccard = 0
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if jaccard_sim(tokens[i], tokens[j]) >= JACCARD_THRESHOLD:
                uf.union(i, j)
                n_jaccard += 1
    group_of_root = {}
    for i, r in enumerate(pool):
        root = uf.find(i)
        group_of_root.setdefault(root, f"G{len(group_of_root) + 1:04d}")
        r["group_identifier"] = group_of_root[root]
    # excluded duplicates point at the group of the file they duplicate
    group_by_hash = {r["norm_hash"]: r["group_identifier"] for r in pool}
    for r in cgt:
        if not r["included"]:
            r["group_identifier"] = group_by_hash.get(r["norm_hash"], "")

    # ---- write files ----------------------------------------------------------
    for d in (SYNTHETIC_DIR, EXPANDED_DIR):
        if d.exists(): shutil.rmtree(d)
        for cat in CANONICAL_LABELS: (d / cat).mkdir(parents=True)

    for r in synth:
        shutil.copy2(r["source_path"], SYNTHETIC_DIR / r["category"] / r["dest_name"])
    with (SYNTHETIC_DIR / "synthetic_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["contract_identifier", "category", "solidifi_folder",
                                          "base_contract", "source_hashes", "group_identifier"])
        w.writeheader()
        for r in synth:
            w.writerow({"contract_identifier": r["dest_name"], "category": r["category"],
                        "solidifi_folder": r["original_label"], "base_contract": r["base_contract"],
                        "source_hashes": r["norm_hash"], "group_identifier": r["group_identifier"]})

    rows = []
    for r in sb + cgt:
        if r["included"]:
            shutil.copy2(r["source_path"], EXPANDED_DIR / r["category"] / r["dest_name"])
        rows.append({
            "original_dataset": r["source_dataset"],
            "dataset_version": r["dataset_version"],
            "source_URL": r["source_url"],
            "contract_identifier": r["dest_name"],
            "category": r["category"],
            "label_set": r["label_set"],
            "vulnerability_label": r["original_label"],
            "DASP/SWC_mapping": r["mapping"],
            "source_hashes": r["norm_hash"],
            "group_identifier": r["group_identifier"],
            "inclusion_decision": "Included" if r["included"] else f"Excluded ({r['exclusion_reason']})",
        })
    with (EXPANDED_DIR / "reproducibility_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- report ---------------------------------------------------------------
    inc_rw = [r for r in sb + cgt if r["included"]]
    sf_groups = {r["group_identifier"] for r in synth}
    rw_groups = {r["group_identifier"] for r in inc_rw}
    print(f"  [SYNTHETIC]  {len(synth)} SolidiFI contracts from "
          f"{len({r['base_contract'] for r in synth})} base contracts -> {len(sf_groups)} groups")
    print(f"  [REAL-WORLD] SmartBugs Curated {len(sb)}; CGT candidate files {len(cgt)} -> "
          f"{sum(r['included'] for r in cgt)} included "
          f"({dict(Counter(r['exclusion_reason'] for r in cgt if not r['included']))} excluded); "
          f"multi-label policy = {cgt_multilabel}")
    print(f"  [GROUPS]     {len(sf_groups | rw_groups)} total "
          f"(SolidiFI {len(sf_groups)}, real-world {len(rw_groups)}, shared {len(sf_groups & rw_groups)}); "
          f"Jaccard links {n_jaccard}")
    print(f"  {'class':<28}{'SolidiFI':>9}{'SmartBugs':>10}{'CGT':>6}{'Total':>7}")
    for cat in CANONICAL_LABELS:
        a = sum(r["category"] == cat for r in synth)
        b = sum(r["category"] == cat for r in sb)
        c = sum(r["category"] == cat for r in cgt if r["included"])
        print(f"  {cat:<28}{a:>9}{b:>10}{c:>6}{a + b + c:>7}")
    print(f"  {'TOTAL':<28}{len(synth):>9}{len(sb):>10}{sum(r['included'] for r in cgt):>6}"
          f"{len(synth) + len(inc_rw):>7}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the UnifiedBench dataset.")
    ap.add_argument("--cgt-multilabel", choices=["drop", "rarest"], default="drop",
                    help="what to do with CGT contracts asserting more than one canonical class")
    args = ap.parse_args()
    print("======================================================")
    print("  UnifiedBench dataset construction")
    print("======================================================")
    build_datasets(args.cgt_multilabel)
