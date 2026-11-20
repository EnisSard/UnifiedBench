"""
reviewer_analyses.py
--------------------
Additional analyses for the revision, on top of analyze_results.py. Reads Run A
(results/runA: repeats, seed_study, ablation_singleton_groups) and Run B
(results/runB/repeats) and writes results/runA/analysis/reviewer_report.txt and
reviewer_analysis.json.

    UB_PROFILE=runA python reviewer_analyses.py

Sections
  1. Label-confidence tiers (rule-based, see TIER RULES below) and a tier-stratified
     sensitivity analysis.
  2. Real-world per-class F1 with support and clone-group bootstrap 95% CIs.
  3. SmartBugs Curated vs CGT: length, class mix, CGT source campaign, accuracy.
  4. Input length: real-world contracts that fit one 512-token window vs longer ones.
  5. Alternative aggregations for the SmartBERT comparisons (fold-level corrected t,
     repeat-level paired t / Wilcoxon, real-world fold-level corrected t) and the
     per-repeat paired real-world differences.
  6. Grouping ablation (clone families vs file-level SolidiFI groups).
  7. Run B (tuned + multi-label) summary with the always-Arithmetic baseline.

TIER RULES (applied per contract by script; no manual judgement)
  Documentation (D): the vulnerable location is documented -- SolidiFI (bug injected
      by the tool at a logged location) or SmartBugs Curated (curator
      @vulnerable_at_lines annotation in the original file). CGT labels carry no
      location.
  Consistency (C): no conflicting assertion. For CGT, a conflict is another source
      dataset in CGT's consolidated table stating that a property mapping to the SAME
      canonical class does NOT hold for the same file (property_holds = f).
  Interpretation (I): the canonical class follows from the source label without
      reinterpretation. SWC-108 (state-variable default visibility) mapped to
      access_control is flagged as an interpretive mapping.
  Corroboration (E): number of independent CGT source datasets asserting the class.

  Tier 1 (High)     D and C and not I
  Tier 2 (Moderate) not D, C and not I   (CGT label from a curated dataset, no
                    location; split into E = 1 and E >= 2 in the report)
  Tier 3 (Low)      a conflict across sources (not C) or an interpretive mapping (I);
                    in Run B also every multi-label contract
"""
import csv
import itertools
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

os.environ.setdefault("UB_PROFILE", "runA")
from analyze_results import (Data, load_run_dir, oof_predictions, bootstrap_weights, cm_of,
                             macro_f1, macro_f1_from_cm, per_class_f1_from_cm, corrected_t, holm,
                             fmt_p, MODEL_ORDER, MODEL_NAME, K)
from build_unified_dataset import SWC_MAP
from train.config import LABELS, DATA_DIR, MANIFEST

ROOT = Path(__file__).parent.resolve()
RA = ROOT / "results" / "runA"
RB = ROOT / "results" / "runB"
OUT = RA / "analysis"
N_BOOT = 2000
INTERPRETIVE_SWC = {108}
SHORT = {m: MODEL_NAME[m].split(" (")[0] for m in MODEL_ORDER}

lines_out = []


def say(s=""):
    print(s)
    lines_out.append(s)


def header(t):
    say("")
    say("=" * 100)
    say("  " + t)
    say("=" * 100)


# ── 1. label-confidence tiers ────────────────────────────────────────────────

def cgt_evidence():
    """fp -> {"pos": {class: {datasets}}, "neg": {class: {datasets}}, "swcs": {class: {swc}}}"""
    ev = defaultdict(lambda: {"pos": defaultdict(set), "neg": defaultdict(set), "swcs": defaultdict(set)})
    with (DATA_DIR / "CGT" / "consolidated.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            try:
                swc = int(row.get("swc", ""))
            except ValueError:
                continue
            if swc not in SWC_MAP or row.get("dataset", "").lower() == "solidifi":
                continue
            fp = row.get("fp_sol2") or row.get("fp_sol")
            cls = SWC_MAP[swc]
            if row.get("property_holds", "").lower() == "t":
                ev[fp]["pos"][cls].add(row["dataset"])
                ev[fp]["swcs"][cls].add(swc)
            elif row.get("property_holds", "").lower() == "f":
                ev[fp]["neg"][cls].add(row["dataset"])
    return ev


def smartbugs_annotated():
    out = {}
    for p in (DATA_DIR / "smartbugs-curated" / "dataset").rglob("*.sol"):
        out[p.name] = "@vulnerable_at_lines" in p.read_text(encoding="utf-8", errors="ignore")
    return out


def assign_tiers(names, labels, sub):
    ev = cgt_evidence()
    sb = smartbugs_annotated()
    tiers, info = [], []
    for n, y, s in zip(names, labels, sub):
        cls = LABELS[y]
        if s == "solidifi":
            d, c, i, e = True, True, False, 0
        elif s == "smartbugs":
            raw = n.split("__", 1)[1]
            d, c, i, e = sb.get(raw, False), True, False, 0
        else:
            fp = n.split("__", 1)[1].rsplit(".", 1)[0]
            r = ev.get(fp)
            pos = r["pos"].get(cls, set()) if r else set()
            neg = (r["neg"].get(cls, set()) - pos) if r else set()
            swcs = r["swcs"].get(cls, set()) if r else set()
            d, c = False, not neg
            i = bool(swcs) and swcs <= INTERPRETIVE_SWC
            e = len(pos)
        t = 1 if (d and c and not i) else (2 if (c and not i) else 3)
        tiers.append(t)
        info.append(dict(doc=d, consistent=c, interpretive=i, n_sources=e))
    return np.array(tiers), info


def tier_section(data, names, oof, models, W):
    header("1. LABEL-CONFIDENCE TIERS (rules in the script header) and tier-stratified results (Run A)")
    tiers, info = assign_tiers(names, data.y, data.sub)
    n_sources = np.array([x["n_sources"] for x in info])
    say(f"  {'subset':<12}{'Tier 1':>8}{'Tier 2':>8}{'Tier 3':>8}   Tier 2 corroborated by >=2 CGT sources")
    for s in ("solidifi", "smartbugs", "cgt"):
        m = data.sub == s
        say(f"  {s:<12}" + "".join(f"{int(((tiers == t) & m).sum()):>8}" for t in (1, 2, 3))
            + f"   {int(((tiers == 2) & m & (n_sources >= 2)).sum())}")
    rw = data.masks["realworld"]
    reasons = Counter()
    for k in np.where(rw & (tiers == 3))[0]:
        reasons["conflict" if not info[k]["consistent"] else "interpretive SWC mapping"] += 1
    say(f"  Tier 3 reasons (real-world): {dict(reasons)}")
    say(f"  real-world tiers by class:")
    for c in range(K):
        m = rw & (data.y == c)
        say(f"    {LABELS[c]:<28}" + "  ".join(f"T{t} {int(((tiers == t) & m).sum()):>3}" for t in (1, 2, 3)))

    say("")
    say("  Real-world accuracy and macro-F1 by tier (out-of-fold, mean over 10 repeats); sensitivity = real-world")
    say("  macro-F1 without Tier 3, and Tier 1 only, with clone-group bootstrap 95% CI")
    say(f"  {'model':<16}{'acc T1':>8}{'acc T2':>8}{'acc T3':>8}{'mF1 all RW':>12}{'mF1 RW w/o T3':>22}{'mF1 RW T1 only':>22}")
    res = {}
    for m in models:
        row = {}
        for t in (1, 2, 3):
            mk = rw & (tiers == t)
            row[f"acc_t{t}"] = float(np.mean([(p[mk] == data.y[mk]).mean() for p in oof[m]])) if mk.sum() else float("nan")
        for key, mk in (("all", rw), ("no_t3", rw & (tiers < 3)), ("t1", rw & (tiers == 1))):
            vals = [macro_f1(data.y[mk], p[mk]) for p in oof[m]]
            boot = np.zeros(W.shape[0])
            for p in oof[m]:
                codes = data.y[mk] * K + p[mk]
                oh = np.zeros((mk.sum(), K * K)); oh[np.arange(mk.sum()), codes] = 1
                boot += macro_f1_from_cm((W[:, mk] @ oh).reshape(-1, K, K))
            boot /= len(oof[m])
            row[f"mf1_{key}"] = [float(np.mean(vals)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        res[m] = row
        f = lambda v: f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]"
        say(f"  {SHORT[m]:<16}{row['acc_t1']:>8.3f}{row['acc_t2']:>8.3f}{row['acc_t3']:>8.3f}{row['mf1_all'][0]:>12.3f}"
            f"{f(row['mf1_no_t3']):>22}{f(row['mf1_t1']):>22}")
    ranks = {k: [m for m, _ in sorted(res.items(), key=lambda kv: -kv[1][f"mf1_{k}"][0])] for k in ("all", "no_t3", "t1")}
    say(f"  ranking all RW : {[SHORT[m] for m in ranks['all']]}")
    say(f"  ranking w/o T3 : {[SHORT[m] for m in ranks['no_t3']]}")
    say(f"  ranking T1 only: {[SHORT[m] for m in ranks['t1']]}")
    say(f"  Spearman rank correlation (all vs w/o T3): "
        f"{stats.spearmanr([ranks['all'].index(m) for m in models], [ranks['no_t3'].index(m) for m in models])[0]:.3f}; "
        f"(all vs T1 only): {stats.spearmanr([ranks['all'].index(m) for m in models], [ranks['t1'].index(m) for m in models])[0]:.3f}")
    rows = [dict(contract_identifier=n, tier=int(t), **i) for n, t, i in zip(names, tiers, info)]
    with (OUT / "label_confidence_tiers.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return {"counts": {s: {t: int(((tiers == t) & (data.sub == s)).sum()) for t in (1, 2, 3)} for s in ("solidifi", "smartbugs", "cgt")},
            "by_model": res, "rankings": ranks}, tiers


# ── 2. per-class real-world F1 with CIs ───────────────────────────────────────

def per_class_section(data, oof, models, W):
    header("2. REAL-WORLD PER-CLASS F1 with support and clone-group bootstrap 95% CI (Run A, mean over repeats)")
    rw = data.masks["realworld"]
    sup = [int(((data.y == c) & rw).sum()) for c in range(K)]
    sup_sb = [int(((data.y == c) & data.masks["smartbugs"]).sum()) for c in range(K)]
    sup_cg = [int(((data.y == c) & data.masks["cgt"]).sum()) for c in range(K)]
    say("  support (real-world = SmartBugs + CGT): " + ", ".join(f"{LABELS[c]} {sup[c]} ({sup_sb[c]}+{sup_cg[c]})" for c in range(K)))
    out = {}
    say(f"  {'model':<16}" + "".join(f"{LABELS[c][:14]:>22}" for c in range(K)))
    for m in models:
        boot = np.zeros((W.shape[0], K))
        pt = np.zeros(K)
        for p in oof[m]:
            codes = data.y[rw] * K + p[rw]
            oh = np.zeros((rw.sum(), K * K)); oh[np.arange(rw.sum()), codes] = 1
            boot += per_class_f1_from_cm((W[:, rw] @ oh).reshape(-1, K, K))
            pt += per_class_f1_from_cm(cm_of(data.y[rw], p[rw]))
        boot /= len(oof[m]); pt /= len(oof[m])
        lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
        out[m] = {LABELS[c]: [float(pt[c]), float(lo[c]), float(hi[c])] for c in range(K)}
        say(f"  {SHORT[m]:<16}" + "".join(f"{f'{pt[c]:.2f} [{lo[c]:.2f},{hi[c]:.2f}]':>22}" for c in range(K)))
    say(f"  note: time_manipulation has {sup[LABELS.index('time_manipulation')]} real-world contracts; its CI spans most of [0,1].")
    return {"support": dict(zip(LABELS, sup)), "f1": out}


# ── 3. SmartBugs vs CGT ───────────────────────────────────────────────────────

def sb_cgt_section(data, names, texts, ntok, oof, models):
    header("3. SMARTBUGS CURATED vs CGT: length, class mix, CGT campaign, accuracy (Run A)")
    man = {r["contract_identifier"]: r for r in csv.DictReader(MANIFEST.open(encoding="utf-8"))}
    loc = np.array([sum(1 for ln in t.splitlines() if ln.strip()) for t in texts])
    out = {}
    for s in ("smartbugs", "cgt"):
        mk = data.masks[s]
        out[s] = {"n": int(mk.sum()), "loc_median": float(np.median(loc[mk])), "loc_iqr": [float(np.percentile(loc[mk], 25)), float(np.percentile(loc[mk], 75))],
                  "tokens_median": float(np.median(ntok[mk])), "share_over_512_tokens": float((ntok[mk] > 510).mean()),
                  "share_over_6_chunks": float((ntok[mk] > 6 * 510).mean()),
                  "classes": {LABELS[c]: int(((data.y == c) & mk).sum()) for c in range(K)}}
        say(f"  {s:<10} n={out[s]['n']:<4} non-empty LOC median {out[s]['loc_median']:.0f} IQR {out[s]['loc_iqr'][0]:.0f}-{out[s]['loc_iqr'][1]:.0f}; "
            f"SmartBERT tokens median {out[s]['tokens_median']:.0f}; >512 tokens {100*out[s]['share_over_512_tokens']:.0f}%; "
            f">6 chunks {100*out[s]['share_over_6_chunks']:.0f}%")
        say(f"             classes {out[s]['classes']}")
    camp = np.array([man[n]["original_dataset"].replace("CGT (", "").rstrip(")") if n in man else "" for n in names], dtype=object)
    cg = data.masks["cgt"]
    campaigns = [c for c, _ in Counter(camp[cg]).most_common()]
    say("")
    say("  accuracy by CGT source campaign (contracts asserted by several campaigns are listed under their combination)")
    say(f"  {'campaign':<28}{'n':>4}  classes" + " " * 40 + "".join(f"{SHORT[m][:9]:>10}" for m in models))
    out["campaigns"] = {}
    for c in campaigns:
        mk = cg & (camp == c)
        cls = dict(Counter(LABELS[y][:6] for y in data.y[mk]))
        accs = {m: float(np.mean([(p[mk] == data.y[mk]).mean() for p in oof[m]])) for m in models}
        out["campaigns"][c] = {"n": int(mk.sum()), "classes": cls, "accuracy": accs}
        say(f"  {c:<28}{int(mk.sum()):>4}  {str(cls):<46}" + "".join(f"{accs[m]:>10.2f}" for m in models))
    say("")
    say("  accuracy by source and class (SmartBERT / XGBoost), mean over repeats")
    for s in ("smartbugs", "cgt"):
        for c in range(K):
            mk = data.masks[s] & (data.y == c)
            if mk.sum() == 0:
                continue
            a = [float(np.mean([(p[mk] == c).mean() for p in oof[m]])) for m in ("smartbert", "xgboost")]
            say(f"    {s:<10}{LABELS[c]:<28} n={int(mk.sum()):>3}   SmartBERT {a[0]:.2f}   XGBoost {a[1]:.2f}")
    return out


# ── 4. length / truncation ────────────────────────────────────────────────────

def length_section(data, ntok, oof, models, W):
    header("4. INPUT LENGTH: real-world contracts within one 512-token window vs longer (Run A, chunked inputs)")
    rw = data.masks["realworld"]
    buckets = [("<=512 tok (1 window)", ntok <= 510), ("513-3060 (2-6 chunks)", (ntok > 510) & (ntok <= 3060)),
               (">3060 (chunk selection)", ntok > 3060)]
    say("  " + "; ".join(f"{b}: n={int((mk & rw).sum())}" for b, mk in buckets))
    say(f"  {'model':<16}" + "".join(f"{b:>26}" for b, _ in buckets) + "      (accuracy | macro-F1)")
    out = {}
    for m in models:
        row = {}
        for b, mk in buckets:
            mk = mk & rw
            acc = float(np.mean([(p[mk] == data.y[mk]).mean() for p in oof[m]]))
            f1 = float(np.mean([macro_f1(data.y[mk], p[mk]) for p in oof[m]]))
            row[b] = [acc, f1]
        out[m] = row
        say(f"  {SHORT[m]:<16}" + "".join(f"{f'{row[b][0]:.3f} | {row[b][1]:.3f}':>26}" for b, _ in buckets))
    say("  class mix per bucket: " + "; ".join(f"{b}: {dict(Counter(LABELS[y][:6] for y in data.y[mk & rw]))}" for b, mk in buckets))
    return {"buckets": {b: int((mk & rw).sum()) for b, mk in buckets}, "by_model": out}


# ── 5. alternative aggregations ───────────────────────────────────────────────

def aggregation_section(repeats, data, oof, models, ref="smartbert"):
    header(f"5. ALTERNATIVE AGGREGATIONS for {SHORT[ref]} vs every model (Run A, 10 repeats x 5 folds)")
    seeds = sorted(repeats)
    ratio = float(np.mean([f["n_test"] / f["n_train"] for s in seeds for f in repeats[s][ref]["folds"]]))
    fold = {m: np.array([f["macro_f1"] for s in seeds for f in repeats[s][m]["folds"]]) for m in models}
    fold_rw = {m: np.array([f["realworld_macro_f1"] for s in seeds for f in repeats[s][m]["folds"]]) for m in models}
    rep = {m: np.array([np.mean([f["macro_f1"] for f in repeats[s][m]["folds"]]) for s in seeds]) for m in models}
    rw = data.masks["realworld"]
    pooled = {m: np.array([macro_f1(data.y[rw], p[rw]) for p in oof[m]]) for m in models}
    others = [m for m in models if m != ref]
    raw = defaultdict(dict)
    rows = {}
    for m in others:
        d_f, d_frw, d_r, d_p = fold[ref] - fold[m], fold_rw[ref] - fold_rw[m], rep[ref] - rep[m], pooled[ref] - pooled[m]
        r = {}
        r["fold_corrected_t"] = corrected_t(d_f, ratio)
        r["fold_rw_corrected_t"] = corrected_t(d_frw, ratio)
        r["repeat_paired_t"] = (float(d_r.mean()), float(stats.ttest_rel(rep[ref], rep[m]).pvalue))
        r["repeat_wilcoxon"] = float(stats.wilcoxon(d_r).pvalue) if np.any(d_r != 0) else 1.0
        r["fold_wilcoxon"] = float(stats.wilcoxon(d_f).pvalue) if np.any(d_f != 0) else 1.0
        r["pooled_rw_per_repeat"] = d_p.tolist()
        r["pooled_rw_wins"] = int((d_p > 0).sum())
        r["pooled_rw_sign_p"] = float(stats.binomtest(int((d_p > 0).sum()), int((d_p != 0).sum())).pvalue) if np.any(d_p != 0) else 1.0
        raw["fold"][m] = r["fold_corrected_t"][4]
        raw["fold_rw"][m] = r["fold_rw_corrected_t"][4]
        raw["rep_t"][m] = r["repeat_paired_t"][1]
        raw["rep_w"][m] = r["repeat_wilcoxon"]
        raw["fold_w"][m] = r["fold_wilcoxon"]
        raw["sign"][m] = r["pooled_rw_sign_p"]
        rows[m] = r
    adj = {k: holm(v) for k, v in raw.items()}
    say("  Holm adjustment over the 7 comparisons with the reference model, separately per test.")
    say("  (a) fold-level overall macro-F1, corrected resampled t (Nadeau-Bengio), n=50")
    say("  (b) fold-level REAL-WORLD macro-F1, corrected resampled t, n=50")
    say("  (c) repeat-level overall macro-F1 (mean of 5 folds), ordinary paired t and Wilcoxon, n=10")
    say("  (d) fold-level overall macro-F1, Wilcoxon signed-rank (ignores fold dependence; anti-conservative)")
    say("  (e) repeat-level POOLED real-world macro-F1: per-repeat paired differences, wins out of 10, sign test")
    say(f"  {'vs':<16}{'(a) diff p_holm':>18}{'(b) diff p_holm':>18}{'(c) t p_holm':>14}{'(c) W p_holm':>14}{'(d) W p_holm':>14}"
        f"{'(e) mean [min,max]':>26}{'wins':>6}{'sign p_holm':>13}")
    for m in others:
        r = rows[m]
        dp = np.array(r["pooled_rw_per_repeat"])
        say(f"  {SHORT[m]:<16}"
            f"{(format(r['fold_corrected_t'][0], '+.3f') + ' ' + fmt_p(adj['fold'][m])):>18}"
            f"{(format(r['fold_rw_corrected_t'][0], '+.3f') + ' ' + fmt_p(adj['fold_rw'][m])):>18}"
            f"{fmt_p(adj['rep_t'][m]):>14}{fmt_p(adj['rep_w'][m]):>14}{fmt_p(adj['fold_w'][m]):>14}"
            f"{(format(dp.mean(), '+.3f') + ' [' + format(dp.min(), '+.3f') + ',' + format(dp.max(), '+.3f') + ']'):>26}"
            f"{r['pooled_rw_wins']:>4}/10{fmt_p(adj['sign'][m]):>13}")
    say("  Note: with 10 repeats the smallest possible two-sided sign-test p is 0.002; after Holm over 7 comparisons")
    say("  only 10/10 wins can reach p_holm < 0.05.")
    return {"rows": rows, "holm": adj, "test_train_ratio": ratio}


# ── 6. ablation ───────────────────────────────────────────────────────────────

def ablation_section(repeats, data, models):
    header("6. GROUPING ABLATION: clone-family groups vs file-level SolidiFI groups (seed 42, Run A)")
    abl = load_run_dir(RA / "ablation_singleton_groups")
    if 42 not in abl:
        say("  (no ablation results)")
        return {}
    rw, sf = data.masks["realworld"], data.masks["solidifi"]
    out = {}
    say(f"  {'model':<16}{'overall grouped':>16}{'overall file-lvl':>17}{'SolidiFI grouped':>17}{'SolidiFI file':>14}{'RW grouped':>12}{'RW file':>9}")
    for m in models:
        if m not in abl[42]:
            continue
        g, f = repeats[42][m], abl[42][m]
        pg, pf = oof_predictions(g, data)[0], oof_predictions(f, data)[0]
        row = [np.mean([x["macro_f1"] for x in g["folds"]]), np.mean([x["macro_f1"] for x in f["folds"]]),
               macro_f1(data.y[sf], pg[sf]), macro_f1(data.y[sf], pf[sf]), macro_f1(data.y[rw], pg[rw]), macro_f1(data.y[rw], pf[rw])]
        out[m] = [float(v) for v in row]
        say(f"  {SHORT[m]:<16}" + "".join(f"{v:>{w}.3f}" for v, w in zip(row, (16, 17, 17, 14, 12, 9))))
    return out


# ── 7. Run B ──────────────────────────────────────────────────────────────────

def runb_section(models):
    header("7. RUN B: nested-CV tuning + 261 multi-label CGT contracts + risk-ranked chunks (3 repeats)")
    rb = load_run_dir(RB / "repeats")
    ra = load_run_dir(RA / "repeats")
    meta = json.loads((RB / "analysis" / "rows.json").read_text(encoding="utf-8"))
    sets = [set(s) for s in meta["label_sets"]]
    names = meta["names"]
    seeds = sorted(rb)
    ar = LABELS.index("arithmetic")
    ml_contracts = {n for n, s in zip(names, sets) if len(s) > 1}
    first = {}
    for i, n in enumerate(names):
        first.setdefault(n, i)
    ml_first = [first[n] for n in ml_contracts]
    base_arith = float(np.mean([ar in sets[i] for i in ml_first]))
    chance = float(np.mean([len(sets[i]) / K for i in ml_first]))
    say(f"  multi-label contracts: {len(ml_contracts)}; mean label-set size {np.mean([len(sets[i]) for i in ml_first]):.2f}; "
        f"always-Arithmetic any-match accuracy {base_arith:.3f}; uniform-random any-match accuracy {chance:.3f}")
    say(f"  seeds {seeds}. 'RW single' = macro-F1 on the 243 single-label real-world contracts (same contracts as Run A).")
    say(f"  {'model':<16}{'Run A RW (same seeds)':>22}{'Run B RW single':>16}{'diff':>8}{'Run B overall':>14}{'Run B SolidiFI':>15}{'ML any-match':>13}  chosen configs (15 folds)")
    out = {}
    for m in models:
        if not all(m in rb[s] for s in seeds):
            continue
        a = np.mean([ra[s][m]["aggregate"]["realworld_macro_f1_mean"] for s in seeds])
        # pooled per repeat, single-label real-world contracts
        b_single, b_all, b_sf, b_ml = [], [], [], []
        for s in seeds:
            yt, yp, src, ml = [], [], [], []
            for f in rb[s][m]["folds"]:
                yt += f["y_true"]; yp += f["y_pred"]; src += f["sources"]; ml += f["multilabel_mask"]
            yt, yp, src, ml = map(np.asarray, (yt, yp, src, ml))
            single = (src == "realworld") & (ml == 0)
            b_single.append(macro_f1(yt[single], yp[single]))
            b_sf.append(macro_f1(yt[src == "solidifi"], yp[src == "solidifi"]))
            b_ml.append(float((yt[ml == 1] == yp[ml == 1]).mean()))
            b_all.append(np.mean([f["macro_f1"] for f in rb[s][m]["folds"]]))
        a_pooled = np.mean([macro_f1(*_pooled_rw(ra[s][m])) for s in seeds])
        cfgs = Counter(json.dumps(f["tuned_hp"], sort_keys=True) for s in seeds for f in rb[s][m]["folds"])
        out[m] = dict(runA_rw_pooled=float(a_pooled), runB_rw_single_pooled=float(np.mean(b_single)),
                      runB_rw_single_per_seed=[float(x) for x in b_single], runB_overall=float(np.mean(b_all)),
                      runB_solidifi=float(np.mean(b_sf)), runB_ml_anymatch=float(np.mean(b_ml)), configs=dict(cfgs))
        cf = ", ".join(f"{k}: {v}" for k, v in cfgs.most_common())
        say(f"  {SHORT[m]:<16}{a_pooled:>22.3f}{np.mean(b_single):>16.3f}{np.mean(b_single) - a_pooled:>+8.3f}{np.mean(b_all):>14.3f}"
            f"{np.mean(b_sf):>15.3f}{np.mean(b_ml):>13.3f}  {cf}")
    say("  Run A RW and Run B RW single are both pooled out-of-fold macro-F1 per repeat, averaged over the same 3 seeds.")
    return {"baseline_always_arithmetic": base_arith, "baseline_random": chance, "by_model": out}


def _pooled_rw(run):
    yt, yp = [], []
    for f in run["folds"]:
        for a, b, s in zip(f["y_true"], f["y_pred"], f["sources"]):
            if s == "realworld":
                yt.append(a); yp.append(b)
    return np.asarray(yt), np.asarray(yp)


def main():
    from transformers import AutoTokenizer
    from train.data_loader import load_records
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    data = Data()
    texts, _, _, _, names = load_records()
    tok = AutoTokenizer.from_pretrained("web3se/SmartBERT-v3")
    ntok = np.array([len(tok(t, add_special_tokens=False, verbose=False)["input_ids"]) for t in texts])
    repeats = load_run_dir(RA / "repeats")
    seeds = sorted(repeats)
    models = [m for m in MODEL_ORDER if all(m in repeats[s] for s in seeds)]
    oof = {m: [oof_predictions(repeats[s][m], data)[0] for s in seeds] for m in models}
    W = bootstrap_weights(data, N_BOOT, rng)
    say(f"Reviewer analyses -- Run A: {len(seeds)} repeats, models {models}; bootstrap B={N_BOOT}")
    R = {}
    R["tiers"], _ = tier_section(data, names, oof, models, W)
    R["per_class_realworld"] = per_class_section(data, oof, models, W)
    R["smartbugs_vs_cgt"] = sb_cgt_section(data, names, texts, ntok, oof, models)
    R["length"] = length_section(data, ntok, oof, models, W)
    R["aggregations"] = aggregation_section(repeats, data, oof, models)
    R["ablation"] = ablation_section(repeats, data, models)
    R["runB"] = runb_section(models)

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    (OUT / "reviewer_analysis.json").write_text(json.dumps(_clean(R), indent=1), encoding="utf-8")
    (OUT / "reviewer_report.txt").write_text("\n".join(lines_out), encoding="utf-8")
    print(f"\n[OK] wrote {OUT / 'reviewer_report.txt'}")


if __name__ == "__main__":
    main()
