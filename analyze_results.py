"""
analyze_results.py
------------------
Statistics for the repeated grouped cross-validation and the multi-seed
leave-one-source-out (LOSO) runs. Reads

    results/repeats/seed_<s>/<model>/fold_results.json      (run_all.py)
    results/loso/seed_<s>/<direction>/<model>/fold_results.json   (cross_source_eval.py)
    results/seed_study/seed_<s>/<model>/fold_results.json   (optional, run_all.py --split-seed)

and writes results/analysis/{report.txt, analysis.json, *.csv}.

Methods
  Fold-level metrics (macro-F1, accuracy, MCC). With R repeats of K-fold CV there are
  n = R*K fold estimates per model. Their 95% CI uses the corrected resampled
  t-statistic of Nadeau & Bengio (2003) as applied to repeated k-fold CV by
  Bouckaert & Frank (2004): variance (1/n + n_test/n_train) * s^2, t with n-1 df.
  Pairwise model comparisons use the same corrected t-test on the paired per-fold
  differences (all models of a repeat share the folds), Holm-adjusted over all pairs.

  Source-level metrics. In every repeat each contract is predicted exactly once
  (out of fold), so per repeat we pool the predictions and compute macro-F1 on a
  source (SolidiFI, real-world, SmartBugs Curated, CGT). Point estimate = mean over
  repeats. Uncertainty = clone-group (cluster) bootstrap: clone-family groups are
  resampled with replacement, separately within SolidiFI and within the real-world
  set, and the repeat-averaged macro-F1 is recomputed on each resample (percentile
  95% CI). Model differences on a source use the SAME resamples for both models
  (paired bootstrap); two-sided bootstrap p-values are Holm-adjusted over all pairs.

  ETH2Vec cannot classify contracts that do not compile; they receive the fold's
  majority training class, so ETH2Vec is scored on the same contracts as every model.
  A matched-subset analysis additionally restricts EVERY model to the compilable
  contracts.
"""
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from train.config import LABELS, NUM_LABELS, RESULTS_DIR
from train.data_loader import load_records, subsource_of

MODEL_ORDER = ["xgboost", "smartembed", "eth2vec", "deepsolid", "hybridvuldetect", "codebert", "smartbert", "gnn"]
MODEL_NAME = {
    "xgboost": "XGBoost (fused)", "smartembed": "SmartEmbed (W2V+RF)", "eth2vec": "ETH2Vec (D2V+SVM)",
    "deepsolid": "DeepSolid", "hybridvuldetect": "HybridVulDetect", "codebert": "CodeBERT",
    "smartbert": "SmartBERT", "gnn": "GNN (Contract-GCN)",
}
SUBSETS = ["all", "solidifi", "realworld", "smartbugs", "cgt"]
K = NUM_LABELS


# ── small helpers ─────────────────────────────────────────────────────────────

def macro_f1_from_cm(cm: np.ndarray) -> np.ndarray:
    """Macro-F1 from confusion matrices of shape (..., K, K), sklearn semantics:
    averaged over the classes present in y_true or y_pred."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support = cm.sum(-1)
    pred = cm.sum(-2)
    denom = support + pred
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    present = denom > 0
    return (f1 * present).sum(-1) / np.maximum(present.sum(-1), 1)


def per_class_f1_from_cm(cm: np.ndarray) -> np.ndarray:
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    denom = cm.sum(-1) + cm.sum(-2)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)


def cm_of(y_true, y_pred, weights=None) -> np.ndarray:
    codes = np.asarray(y_true) * K + np.asarray(y_pred)
    return np.bincount(codes, weights=weights, minlength=K * K).reshape(K, K)


def macro_f1(y_true, y_pred) -> float:
    return float(macro_f1_from_cm(cm_of(y_true, y_pred)))


def corrected_t(values, test_train_ratio):
    """Mean, SD, corrected 95% CI half-width, and the corrected t / p for H0: mean = 0."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
    se = np.sqrt((1.0 / n + test_train_ratio) * sd ** 2)
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    t = mean / se if se > 0 else (np.inf if mean != 0 else 0.0)
    p = float(2 * stats.t.sf(abs(t), n - 1)) if n > 1 else float("nan")
    return mean, sd, float(tcrit * se), float(t), p


def holm(pvals: dict) -> dict:
    keys = sorted(pvals, key=lambda k: pvals[k])
    m, adj, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ── loading ───────────────────────────────────────────────────────────────────

def load_run_dir(root: Path) -> dict:
    """{seed: {model: fold_results}} for every seed_<s> folder under root."""
    out = {}
    for d in sorted(root.glob("seed_*")):
        seed = int(d.name.split("_")[1])
        for m in MODEL_ORDER:
            f = d / m / "fold_results.json"
            if f.exists():
                out.setdefault(seed, {})[m] = json.loads(f.read_text(encoding="utf-8"))
    return out


class Data:
    def __init__(self):
        texts, labels, groups, sources, names = load_records()
        self.y = np.asarray(labels)
        self.n = len(labels)
        self.groups = np.asarray(groups, dtype=object)
        self.source = np.asarray(sources, dtype=object)
        self.sub = np.asarray([subsource_of(n) for n in names], dtype=object)
        self.masks = {
            "all": np.ones(self.n, bool),
            "solidifi": self.source == "solidifi",
            "realworld": self.source == "realworld",
            "smartbugs": self.sub == "smartbugs",
            "cgt": self.sub == "cgt",
        }


def oof_predictions(run: dict, data: Data) -> tuple:
    """Out-of-fold prediction vector (one per contract) + eligibility vector."""
    pred = np.full(data.n, -1, dtype=int)
    elig = np.ones(data.n, dtype=bool)
    for f in run["folds"]:
        idx = np.asarray(f["test_idx"], dtype=int)
        assert np.array_equal(data.y[idx], np.asarray(f["y_true"])), "results do not match the dataset"
        pred[idx] = f["y_pred"]
        if "test_eligible" in f:
            elig[idx] = f["test_eligible"]
    return pred, elig


# ── analyses ──────────────────────────────────────────────────────────────────

def fold_level(runs: dict, models: list, data: Data, restrict: np.ndarray | None = None):
    """Per-model fold metrics (optionally recomputed on a contract subset) and pairwise tests."""
    seeds = sorted(runs)
    per_model = {}
    ratios = []
    for m in models:
        vals = defaultdict(list)
        for s in seeds:
            for f in runs[s][m]["folds"]:
                idx = np.asarray(f["test_idx"])
                yt, yp = np.asarray(f["y_true"]), np.asarray(f["y_pred"])
                if restrict is not None:
                    keep = restrict[idx]
                    yt, yp = yt[keep], yp[keep]
                    vals["macro_f1"].append(macro_f1(yt, yp))
                    vals["accuracy"].append(float((yt == yp).mean()))
                else:
                    vals["macro_f1"].append(f["macro_f1"])
                    vals["accuracy"].append(f["accuracy"])
                    vals["mcc"].append(f["mcc"])
                    vals["train_time_s"].append(f.get("train_time_s", np.nan))
                if m == models[0]:
                    ratios.append(f["n_test"] / max(f["n_train"], 1))
        per_model[m] = vals
    ratio = float(np.mean(ratios))
    n_folds = len(per_model[models[0]]["macro_f1"])

    table = {}
    for m in models:
        row = {}
        for metric, v in per_model[m].items():
            if metric == "train_time_s":
                row[metric] = float(np.nanmean(v))
                continue
            mean, sd, half, _, _ = corrected_t(v, ratio)
            row[metric] = {"mean": mean, "sd": sd, "ci95": [mean - half, mean + half]}
        table[m] = row

    raw_p, pairs = {}, {}
    for a, b in itertools.combinations(models, 2):
        d = np.asarray(per_model[a]["macro_f1"]) - np.asarray(per_model[b]["macro_f1"])
        mean, sd, half, t, p = corrected_t(d, ratio)
        pairs[(a, b)] = {"diff": mean, "ci95": [mean - half, mean + half], "t": t, "p": p}
        raw_p[(a, b)] = p
    for k, p_adj in holm(raw_p).items():
        pairs[k]["p_holm"] = p_adj
    return {"n_folds": n_folds, "test_train_ratio": ratio, "table": table,
            "pairwise_macro_f1": {f"{a}|{b}": v for (a, b), v in pairs.items()},
            "per_fold_macro_f1": {m: per_model[m]["macro_f1"] for m in models}}


def bootstrap_weights(data: Data, n_boot: int, rng, strata=("solidifi", "realworld")) -> np.ndarray:
    """(n_boot, n) contract weights from a clone-group bootstrap stratified by source."""
    W = np.zeros((n_boot, data.n))
    for s in strata:
        mask = data.source == s
        g_ids, inv = np.unique(data.groups[mask], return_inverse=True)
        counts = rng.multinomial(len(g_ids), np.full(len(g_ids), 1 / len(g_ids)), size=n_boot)
        W[:, mask] = counts[:, inv]
    return W


def source_level(oof: dict, models: list, data: Data, W: np.ndarray, restrict=None):
    """Repeat-pooled macro-F1 per source with cluster-bootstrap CIs and paired tests."""
    onehot_cache = {}
    res, boot = {}, {}
    for m in models:
        res[m], boot[m] = {}, {}
        for sub in SUBSETS:
            mask = data.masks[sub] if restrict is None else data.masks[sub] & restrict
            if mask.sum() == 0:
                continue
            per_rep, boot_acc = [], np.zeros(W.shape[0])
            for pred in oof[m]:
                codes = data.y[mask] * K + pred[mask]
                per_rep.append(float(macro_f1_from_cm(cm_of(data.y[mask], pred[mask]))))
                key = codes.tobytes()
                if key not in onehot_cache:
                    oh = np.zeros((mask.sum(), K * K))
                    oh[np.arange(mask.sum()), codes] = 1
                    onehot_cache[key] = oh
                cms = (W[:, mask] @ onehot_cache[key]).reshape(-1, K, K)
                boot_acc += macro_f1_from_cm(cms)
            bvals = boot_acc / len(oof[m])
            boot[m][sub] = bvals
            res[m][sub] = {
                "mean": float(np.mean(per_rep)), "sd_over_repeats": float(np.std(per_rep, ddof=1)) if len(per_rep) > 1 else 0.0,
                "min": float(np.min(per_rep)), "max": float(np.max(per_rep)),
                "per_repeat": per_rep, "n_contracts": int(mask.sum()),
                "ci95_bootstrap": [float(np.percentile(bvals, 2.5)), float(np.percentile(bvals, 97.5))],
            }
        if "solidifi" in boot[m] and "realworld" in boot[m]:
            d = boot[m]["realworld"] - boot[m]["solidifi"]
            res[m]["gap_realworld_minus_solidifi"] = {
                "mean": res[m]["realworld"]["mean"] - res[m]["solidifi"]["mean"],
                "ci95_bootstrap": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            }
    pairwise = {}
    for sub in SUBSETS:
        raw, entries = {}, {}
        for a, b in itertools.combinations(models, 2):
            if sub not in boot[a] or sub not in boot[b]:
                continue
            d = boot[a][sub] - boot[b][sub]
            p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
            entries[(a, b)] = {"diff": res[a][sub]["mean"] - res[b][sub]["mean"],
                               "ci95_bootstrap": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                               "p": p}
            raw[(a, b)] = p
        for k, p_adj in holm(raw).items():
            entries[k]["p_holm"] = p_adj
        pairwise[sub] = {f"{a}|{b}": v for (a, b), v in entries.items()}
    return res, pairwise


def per_class(oof: dict, models: list, data: Data):
    out = {}
    for m in models:
        out[m] = {}
        for sub in ("all", "solidifi", "realworld"):
            mask = data.masks[sub]
            f1s = np.mean([per_class_f1_from_cm(cm_of(data.y[mask], p[mask])) for p in oof[m]], axis=0)
            out[m][sub] = {LABELS[i]: float(f1s[i]) for i in range(K)}
    return out


def confusion(oof: dict, models: list, data: Data):
    return {m: {sub: np.sum([cm_of(data.y[data.masks[sub]], p[data.masks[sub]]) for p in oof[m]], axis=0).astype(int).tolist()
                for sub in ("all", "solidifi", "realworld")} for m in models}


def loso(loso_root: Path, data: Data, n_boot: int, rng):
    out = {}
    seeds = sorted(int(d.name.split("_")[1]) for d in loso_root.glob("seed_*"))
    if not seeds:
        return out
    W = bootstrap_weights(data, n_boot, rng)
    directions = sorted({p.name for s in seeds for p in (loso_root / f"seed_{s}").iterdir() if p.is_dir()})
    for direction in directions:
        for m in MODEL_ORDER:
            preds = []
            for s in seeds:
                f = loso_root / f"seed_{s}" / direction / m / "fold_results.json"
                if f.exists():
                    preds.append(oof_predictions(json.loads(f.read_text(encoding="utf-8")), data)[0])
            if not preds:
                continue
            test_src = direction.split("_to_")[1]
            subs = ["realworld", "smartbugs", "cgt"] if test_src == "realworld" else ["solidifi"]
            entry = {"n_seeds": len(preds), "seeds": seeds}
            for sub in subs:
                mask = data.masks[sub]
                vals = [macro_f1(data.y[mask], p[mask]) for p in preds]
                oh_boot = np.zeros(W.shape[0])
                for p in preds:
                    codes = data.y[mask] * K + p[mask]
                    oh = np.zeros((mask.sum(), K * K)); oh[np.arange(mask.sum()), codes] = 1
                    oh_boot += macro_f1_from_cm((W[:, mask] @ oh).reshape(-1, K, K))
                oh_boot /= len(preds)
                entry[sub] = {"mean": float(np.mean(vals)), "sd_over_seeds": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                              "min": float(np.min(vals)), "max": float(np.max(vals)), "per_seed": vals,
                              "ci95_bootstrap": [float(np.percentile(oh_boot, 2.5)), float(np.percentile(oh_boot, 97.5))]}
            out.setdefault(direction, {})[m] = entry
    return out


def seed_study(root: Path, repeats: dict, data: Data, split_seed: int):
    """Model-seed variability on a FIXED split (split_seed): the repeat with seed == split_seed
    plus every run in results/seed_study/."""
    runs = load_run_dir(root)
    if split_seed in repeats:
        runs.setdefault(split_seed, {}).update({m: r for m, r in repeats[split_seed].items() if m not in runs.get(split_seed, {})})
    out = {}
    for m in MODEL_ORDER:
        seeds = [s for s in sorted(runs) if m in runs[s]]
        if len(seeds) < 2:
            continue
        cv_means = [float(np.mean([f["macro_f1"] for f in runs[s][m]["folds"]])) for s in seeds]
        rw = [macro_f1(data.y[data.masks["realworld"]], oof_predictions(runs[s][m], data)[0][data.masks["realworld"]]) for s in seeds]
        out[m] = {"seeds": seeds, "cv_macro_f1_per_seed": cv_means, "cv_macro_f1_sd": float(np.std(cv_means, ddof=1)),
                  "cv_macro_f1_range": [min(cv_means), max(cv_means)],
                  "realworld_macro_f1_per_seed": rw, "realworld_macro_f1_sd": float(np.std(rw, ddof=1)),
                  "realworld_macro_f1_range": [min(rw), max(rw)]}
    return out


# ── report ────────────────────────────────────────────────────────────────────

def write_report(A: dict, out_dir: Path):
    L = []
    fl = A["fold_level"]
    models = A["models"]
    L.append("=" * 100)
    L.append(f"  REPEATED STRATIFIED GROUPED 5-FOLD CV   repeats={A['n_repeats']} (seeds {A['seeds']})   "
             f"fold estimates per model={fl['n_folds']}   mean n_test/n_train={fl['test_train_ratio']:.3f}")
    L.append("  mean +/- SD over all folds, [95% CI, Nadeau-Bengio corrected t]")
    L.append("=" * 100)
    L.append(f"  {'Model':<22}{'macro-F1':>30}{'accuracy':>30}{'MCC':>30}{'s/fold':>9}")
    for m in models:
        r = fl["table"][m]
        cell = lambda d: f"{d['mean']:.3f}+/-{d['sd']:.3f} [{d['ci95'][0]:.3f},{d['ci95'][1]:.3f}]"
        L.append(f"  {MODEL_NAME[m]:<22}{cell(r['macro_f1']):>30}{cell(r['accuracy']):>30}{cell(r['mcc']):>30}{r['train_time_s']:>9.1f}")
    L.append("")
    L.append("  Pairwise macro-F1: corrected repeated-CV t-test on paired fold differences, Holm-adjusted")
    hdr = f"  {'':<22}" + "".join(f"{MODEL_NAME[m][:10]:>12}" for m in models)
    for title, field in (("difference (row - column)", "diff"), ("Holm-adjusted p", "p_holm")):
        L.append(f"  -- {title}")
        L.append(hdr)
        for a in models:
            row = f"  {MODEL_NAME[a]:<22}"
            for b in models:
                if a == b:
                    row += f"{'--':>12}"
                    continue
                k = f"{a}|{b}" if f"{a}|{b}" in fl["pairwise_macro_f1"] else f"{b}|{a}"
                e = fl["pairwise_macro_f1"][k]
                if field == "diff":
                    sign = 1 if k.startswith(a + "|") else -1
                    row += f"{sign * e['diff']:+.3f}{'*' if e['p_holm'] < 0.05 else ' '}".rjust(12)
                else:
                    row += fmt_p(e["p_holm"]).rjust(12)
            L.append(row)
    L.append("  (* = Holm-adjusted p < 0.05)")
    L.append("")

    sl = A["source_level"]
    L.append("=" * 100)
    L.append("  SOURCE-LEVEL macro-F1 (out-of-fold predictions pooled per repeat, mean over repeats)")
    L.append("  [95% CI: clone-group bootstrap stratified by source]   gap = real-world - SolidiFI")
    L.append("=" * 100)
    L.append(f"  {'Model':<22}" + "".join(f"{s:>22}" for s in SUBSETS) + f"{'gap':>24}")
    for m in models:
        row = f"  {MODEL_NAME[m]:<22}"
        for s in SUBSETS:
            e = sl["by_model"][m][s]
            row += f"{e['mean']:.3f} [{e['ci95_bootstrap'][0]:.3f},{e['ci95_bootstrap'][1]:.3f}]".rjust(22)
        g = sl["by_model"][m]["gap_realworld_minus_solidifi"]
        row += f"{g['mean']:+.3f} [{g['ci95_bootstrap'][0]:+.3f},{g['ci95_bootstrap'][1]:+.3f}]".rjust(24)
        L.append(row)
    L.append(f"  contracts: " + ", ".join(f"{s}={sl['by_model'][models[0]][s]['n_contracts']}" for s in SUBSETS))
    L.append("  SD over repeats: " + "; ".join(
        f"{MODEL_NAME[m]} RW {sl['by_model'][m]['realworld']['sd_over_repeats']:.3f}" for m in models))
    for sub in ("realworld", "smartbugs", "cgt", "solidifi"):
        L.append("")
        L.append(f"  Paired bootstrap on {sub} macro-F1, significant pairs (Holm p < 0.05):")
        sig = [(k, v) for k, v in sl["pairwise"][sub].items() if v["p_holm"] < 0.05]
        if not sig:
            L.append("    none")
        for k, v in sorted(sig, key=lambda kv: -abs(kv[1]["diff"])):
            a, b = k.split("|")
            L.append(f"    {MODEL_NAME[a]} - {MODEL_NAME[b]} = {v['diff']:+.3f}  "
                     f"[{v['ci95_bootstrap'][0]:+.3f},{v['ci95_bootstrap'][1]:+.3f}]  p_holm={fmt_p(v['p_holm'])}")
    L.append("")

    L.append("=" * 100)
    L.append("  PER-CLASS F1 (all contracts; out-of-fold pooled per repeat, mean over repeats)")
    L.append("=" * 100)
    L.append(f"  {'Model':<22}" + "".join(f"{l[:12]:>14}" for l in LABELS))
    for m in models:
        L.append(f"  {MODEL_NAME[m]:<22}" + "".join(f"{A['per_class'][m]['all'][l]:>14.3f}" for l in LABELS))
    L.append("")

    if A.get("matched_eth2vec"):
        me = A["matched_eth2vec"]
        L.append("=" * 100)
        L.append(f"  MATCHED SUBSET: only the {me['n_contracts']} contracts that compile to EVM bytecode "
                 f"(SolidiFI {me['n_solidifi']}, real-world {me['n_realworld']})")
        L.append("=" * 100)
        L.append(f"  {'Model':<22}{'fold macro-F1 [corr. CI]':>34}{'real-world pooled [boot CI]':>32}{'ETH2Vec - model (Holm p)':>30}")
        for m in models:
            r = me["fold_level"]["table"][m]["macro_f1"]
            rw = me["source_level"][m]["realworld"]
            if m == "eth2vec":
                cmp_ = "--"
            else:
                k = f"eth2vec|{m}" if f"eth2vec|{m}" in me["fold_level"]["pairwise_macro_f1"] else f"{m}|eth2vec"
                e = me["fold_level"]["pairwise_macro_f1"][k]
                sign = 1 if k.startswith("eth2vec|") else -1
                cmp_ = f"{sign * e['diff']:+.3f} ({fmt_p(e['p_holm'])})"
            L.append(f"  {MODEL_NAME[m]:<22}"
                     f"{r['mean']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]".rjust(34)
                     + f"{rw['mean']:.3f} [{rw['ci95_bootstrap'][0]:.3f},{rw['ci95_bootstrap'][1]:.3f}]".rjust(32)
                     + cmp_.rjust(30))
        L.append("")

    if A.get("seed_study"):
        L.append("=" * 100)
        L.append(f"  MODEL-SEED STUDY (fold split fixed at seed {A['seed_study_split']}, only the training seed varies)")
        L.append("=" * 100)
        for m, e in A["seed_study"].items():
            L.append(f"  {MODEL_NAME[m]:<22} seeds {e['seeds']}  CV macro-F1 SD {e['cv_macro_f1_sd']:.4f} "
                     f"range [{e['cv_macro_f1_range'][0]:.3f},{e['cv_macro_f1_range'][1]:.3f}]   "
                     f"real-world SD {e['realworld_macro_f1_sd']:.4f} range [{e['realworld_macro_f1_range'][0]:.3f},{e['realworld_macro_f1_range'][1]:.3f}]")
        L.append("")

    if A.get("loso"):
        L.append("=" * 100)
        L.append("  LEAVE-ONE-SOURCE-OUT macro-F1: mean over seeds +/- SD [min,max] {bootstrap 95% CI}")
        L.append("=" * 100)
        for direction, per in A["loso"].items():
            subs = ["realworld", "smartbugs", "cgt"] if direction.endswith("realworld") else ["solidifi"]
            L.append(f"  {direction}")
            for m in models:
                if m not in per:
                    continue
                cells = []
                for s in subs:
                    e = per[m][s]
                    cells.append(f"{s}: {e['mean']:.3f}+/-{e['sd_over_seeds']:.3f} [{e['min']:.3f},{e['max']:.3f}] "
                                 f"{{{e['ci95_bootstrap'][0]:.3f},{e['ci95_bootstrap'][1]:.3f}}}")
                L.append(f"    {MODEL_NAME[m]:<22} n_seeds={per[m]['n_seeds']}  " + "   ".join(cells))
        L.append("")

    (out_dir / "report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def write_csvs(A: dict, out_dir: Path):
    models = A["models"]
    with (out_dir / "main_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model"] + [f"{m}_{k}" for m in ("macro_f1", "accuracy", "mcc") for k in ("mean", "sd", "ci_lo", "ci_hi")]
                   + ["train_time_s"] + [f"{s}_macro_f1_{k}" for s in SUBSETS for k in ("mean", "ci_lo", "ci_hi")]
                   + ["gap_mean", "gap_ci_lo", "gap_ci_hi"])
        for m in models:
            r = A["fold_level"]["table"][m]
            row = [m]
            for met in ("macro_f1", "accuracy", "mcc"):
                row += [r[met]["mean"], r[met]["sd"], *r[met]["ci95"]]
            row.append(r["train_time_s"])
            for s in SUBSETS:
                e = A["source_level"]["by_model"][m][s]
                row += [e["mean"], *e["ci95_bootstrap"]]
            g = A["source_level"]["by_model"][m]["gap_realworld_minus_solidifi"]
            row += [g["mean"], *g["ci95_bootstrap"]]
            w.writerow(row)
    with (out_dir / "pairwise.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["analysis", "model_a", "model_b", "diff_a_minus_b", "ci_lo", "ci_hi", "p", "p_holm"])
        for k, v in A["fold_level"]["pairwise_macro_f1"].items():
            w.writerow(["cv_fold_macro_f1", *k.split("|"), v["diff"], *v["ci95"], v["p"], v["p_holm"]])
        for sub, entries in A["source_level"]["pairwise"].items():
            for k, v in entries.items():
                w.writerow([f"pooled_{sub}_macro_f1", *k.split("|"), v["diff"], *v["ci95_bootstrap"], v["p"], v["p_holm"]])


def main():
    ap = argparse.ArgumentParser(description="Statistics for the repeated-CV and LOSO results.")
    ap.add_argument("--results", default=str(RESULTS_DIR))
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--split-seed", type=int, default=42, help="fixed split used by the seed study")
    a = ap.parse_args()
    root = Path(a.results)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)

    data = Data()
    repeats = load_run_dir(root / "repeats")
    seeds = sorted(repeats)
    models = [m for m in MODEL_ORDER if all(m in repeats[s] for s in seeds)]
    missing = sorted({m for s in seeds for m in MODEL_ORDER if m not in repeats[s]})
    if missing:
        print(f"[WARN] models missing in some repeats (left out): {missing}")
    print(f"[ANALYSIS] repeats: {seeds}; models: {models}")

    oof = {m: [oof_predictions(repeats[s][m], data)[0] for s in seeds] for m in models}
    for m in models:
        assert all((p >= 0).all() for p in oof[m]), f"{m}: some contracts were never predicted"

    A = {"n_repeats": len(seeds), "seeds": seeds, "models": models}
    A["fold_level"] = fold_level(repeats, models, data)
    W = bootstrap_weights(data, a.n_boot, rng)
    by_model, pairwise = source_level(oof, models, data, W)
    A["source_level"] = {"n_boot": a.n_boot, "by_model": by_model, "pairwise": pairwise}
    A["per_class"] = per_class(oof, models, data)
    A["confusion_matrices_summed_over_repeats"] = confusion(oof, models, data)

    if "eth2vec" in models:
        elig = oof_predictions(repeats[seeds[0]]["eth2vec"], data)[1]
        by_m, pw = source_level(oof, models, data, W, restrict=elig)
        A["matched_eth2vec"] = {
            "n_contracts": int(elig.sum()),
            "n_solidifi": int((elig & data.masks["solidifi"]).sum()),
            "n_realworld": int((elig & data.masks["realworld"]).sum()),
            "fold_level": fold_level(repeats, models, data, restrict=elig),
            "source_level": by_m, "pairwise": pw,
        }

    if (root / "seed_study").exists():
        A["seed_study"] = seed_study(root / "seed_study", repeats, data, a.split_seed)
        A["seed_study_split"] = a.split_seed
    if (root / "loso").exists():
        A["loso"] = loso(root / "loso", data, a.n_boot, rng)

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    (out_dir / "analysis.json").write_text(json.dumps(_clean(A), indent=1), encoding="utf-8")
    write_csvs(A, out_dir)
    write_report(A, out_dir)
    print(f"\n[OK] wrote {out_dir / 'report.txt'}, analysis.json, main_table.csv, pairwise.csv")


if __name__ == "__main__":
    main()
