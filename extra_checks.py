"""
extra_checks.py
---------------
Three follow-up analyses on the saved Run A predictions (no training):

  A. CGT-only per-class F1 with support and clone-group bootstrap 95% CI (Reviewer 5, Q1).
  B. Sensitivity of the corrected resampled t-test to the test/train ratio (Reviewer 4, point 3).
     Grouped folds differ in size, so the Nadeau-Bengio correction is recomputed with the
     mean, the median and the LARGEST per-fold n_test/n_train ratio (largest = most
     conservative), and with fold differences weighted by test-set size.
  C. Majority-vote ensemble (supplementary). Members are chosen by the fold-level CV
     macro-F1 ranking, not by the real-world score, so the real-world evaluation is not used
     for selection: top-3 (SmartBERT, CodeBERT, XGBoost) and top-5. Ties go to the
     highest-ranked member. Compared with SmartBERT by a paired clone-group bootstrap.

    UB_PROFILE=runA python extra_checks.py
Writes results/runA/analysis/extra_checks.txt and extra_checks.json.
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

os.environ.setdefault("UB_PROFILE", "runA")
from analyze_results import (Data, load_run_dir, oof_predictions, bootstrap_weights, cm_of,
                             macro_f1, macro_f1_from_cm, per_class_f1_from_cm, corrected_t, holm,
                             fmt_p, MODEL_ORDER, MODEL_NAME, K)
from train.config import LABELS

ROOT = Path(__file__).parent.resolve()
RA = ROOT / "results" / "runA"
OUT = RA / "analysis"
N_BOOT = 2000
SHORT = {m: MODEL_NAME[m].split(" (")[0] for m in MODEL_ORDER}
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def header(t):
    say("")
    say("=" * 100)
    say("  " + t)
    say("=" * 100)


def boot_f1(W, y, p, mask, per_class=False):
    codes = y[mask] * K + p[mask]
    oh = np.zeros((mask.sum(), K * K))
    oh[np.arange(mask.sum()), codes] = 1
    cms = (W[:, mask] @ oh).reshape(-1, K, K)
    return per_class_f1_from_cm(cms) if per_class else macro_f1_from_cm(cms)


def cgt_per_class(data, oof, models, W):
    header("A. CGT-ONLY PER-CLASS F1 with support and clone-group bootstrap 95% CI (Run A, mean over 10 repeats)")
    cg = data.masks["cgt"]
    sup = [int(((data.y == c) & cg).sum()) for c in range(K)]
    say("  CGT support: " + ", ".join(f"{LABELS[c]} {sup[c]}" for c in range(K)))
    say(f"  {'model':<16}" + "".join(f"{LABELS[c][:14]:>22}" for c in range(K)) + f"{'macro-F1':>22}")
    out = {}
    for m in models:
        bpc, pc, bm, pm = np.zeros((W.shape[0], K)), np.zeros(K), np.zeros(W.shape[0]), 0.0
        for p in oof[m]:
            bpc += boot_f1(W, data.y, p, cg, per_class=True)
            bm += boot_f1(W, data.y, p, cg)
            pc += per_class_f1_from_cm(cm_of(data.y[cg], p[cg]))
            pm += macro_f1(data.y[cg], p[cg])
        r = len(oof[m])
        bpc, pc, bm, pm = bpc / r, pc / r, bm / r, pm / r
        lo, hi = np.percentile(bpc, 2.5, axis=0), np.percentile(bpc, 97.5, axis=0)
        mlo, mhi = np.percentile(bm, [2.5, 97.5])
        out[m] = {"per_class": {LABELS[c]: [float(pc[c]), float(lo[c]), float(hi[c])] for c in range(K)},
                  "macro_f1": [float(pm), float(mlo), float(mhi)]}
        say(f"  {SHORT[m]:<16}" + "".join(f"{f'{pc[c]:.2f} [{lo[c]:.2f},{hi[c]:.2f}]':>22}" for c in range(K))
            + f"{f'{pm:.3f} [{mlo:.3f},{mhi:.3f}]':>22}")
    say("  note: CGT has only 3 Access Control and 1 Time Manipulation contract; those per-class values carry")
    say("  no information and CGT macro-F1 is driven by the four classes with >= 16 contracts.")
    return {"support": dict(zip(LABELS, sup)), "models": out}


def ratio_sensitivity(repeats, models, ref="smartbert"):
    header("B. CORRECTED RESAMPLED t-TEST: sensitivity to the test/train ratio (SmartBERT vs each model, n=50 folds)")
    seeds = sorted(repeats)
    folds = [f for s in seeds for f in repeats[s][ref]["folds"]]
    n_te = np.array([f["n_test"] for f in folds], float)
    n_tr = np.array([f["n_train"] for f in folds], float)
    r = n_te / n_tr
    variants = {"mean": float(r.mean()), "median": float(np.median(r)), "max": float(r.max())}
    say(f"  fold test sizes: min {int(n_te.min())}, median {int(np.median(n_te))}, max {int(n_te.max())}; "
        f"n_test/n_train: min {r.min():.3f}, mean {r.mean():.3f}, median {np.median(r):.3f}, max {r.max():.3f}")
    say("  Each cell: difference, Holm-adjusted p (over the 7 comparisons). 'weighted' = fold differences weighted")
    say("  by n_test (weighted mean and weighted SD) with the mean ratio.")
    out = {}
    others = [m for m in models if m != ref]
    for metric in ("macro_f1", "realworld_macro_f1"):
        say(f"  metric: {metric}")
        say(f"  {'vs':<16}" + "".join(f"{f'ratio={k} ({v:.3f})':>24}" for k, v in variants.items()) + f"{'weighted (mean ratio)':>24}")
        raw = {k: {} for k in list(variants) + ["weighted"]}
        diff = {}
        for m in others:
            d = np.array([fa[metric] - fb[metric] for s in seeds
                          for fa, fb in zip(repeats[s][ref]["folds"], repeats[s][m]["folds"])])
            diff[m] = float(d.mean())
            for k, v in variants.items():
                raw[k][m] = corrected_t(d, v)[4]
            w = n_te / n_te.sum()
            mu = float((w * d).sum())
            var = float((w * (d - mu) ** 2).sum() * len(d) / (len(d) - 1))
            se = np.sqrt((1 / len(d) + variants["mean"]) * var)
            raw["weighted"][m] = float(2 * stats.t.sf(abs(mu / se), len(d) - 1))
            diff[m + "_w"] = mu
        adj = {k: holm(v) for k, v in raw.items()}
        for m in others:
            cells = [f"{diff[m]:+.3f} {fmt_p(adj[k][m])}" for k in variants] + [f"{diff[m + '_w']:+.3f} {fmt_p(adj['weighted'][m])}"]
            say(f"  {SHORT[m]:<16}" + "".join(f"{c:>24}" for c in cells))
        sig = {k: sorted(SHORT[m] for m in others if adj[k][m] < 0.05) for k in adj}
        same = len({tuple(v) for v in sig.values()}) == 1
        say(f"  significant at p_holm < 0.05: {sig}")
        say(f"  -> conclusions {'IDENTICAL' if same else 'DIFFER'} across ratio choices")
        out[metric] = {"ratios": variants, "p_raw": raw, "p_holm": adj, "significant": sig, "identical": same}
    return out


def ensemble(repeats, data, oof, models, W):
    header("C. MAJORITY-VOTE ENSEMBLE (members ranked by fold-level CV macro-F1, not by real-world score)")
    seeds = sorted(repeats)
    cv = {m: np.mean([f["macro_f1"] for s in seeds for f in repeats[s][m]["folds"]]) for m in models}
    rank = sorted(models, key=lambda m: -cv[m])
    say("  CV ranking: " + ", ".join(f"{SHORT[m]} {cv[m]:.3f}" for m in rank))
    rw = data.masks["realworld"]
    res = {}
    base = [boot_f1(W, data.y, p, rw) for p in oof["smartbert"]]
    for k in (3, 5):
        mem = rank[:k]
        preds = []
        for i in range(len(seeds)):
            votes = np.zeros((data.n, K))
            for j, m in enumerate(mem):
                votes[np.arange(data.n), oof[m][i]] += 1 + 1e-3 * (k - j)  # ties -> higher-ranked member
            preds.append(votes.argmax(1))
        row = {}
        for s in ("solidifi", "realworld", "smartbugs", "cgt"):
            mk = data.masks[s]
            row[s] = float(np.mean([macro_f1(data.y[mk], p[mk]) for p in preds]))
        b = np.mean([boot_f1(W, data.y, p, rw) for p in preds], axis=0)
        d = b - np.mean(base, axis=0)
        sb = float(np.mean([macro_f1(data.y[rw], p[rw]) for p in oof["smartbert"]]))
        row.update(members=[SHORT[m] for m in mem], realworld_ci=[float(x) for x in np.percentile(b, [2.5, 97.5])],
                   diff_vs_smartbert=float(row["realworld"] - sb), diff_ci=[float(x) for x in np.percentile(d, [2.5, 97.5])],
                   p_boot=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                   wins=int(sum(macro_f1(data.y[rw], pe[rw]) > macro_f1(data.y[rw], ps[rw]) for pe, ps in zip(preds, oof["smartbert"]))))
        res[f"top{k}"] = row
        say(f"  top-{k} vote {row['members']}")
        say(f"    SolidiFI {row['solidifi']:.3f}  real-world {row['realworld']:.3f} [{row['realworld_ci'][0]:.3f}, {row['realworld_ci'][1]:.3f}]"
            f"  SmartBugs {row['smartbugs']:.3f}  CGT {row['cgt']:.3f}")
        say(f"    vs SmartBERT alone (real-world {sb:.3f}): {row['diff_vs_smartbert']:+.3f} "
            f"[{row['diff_ci'][0]:+.3f}, {row['diff_ci'][1]:+.3f}], paired bootstrap p={row['p_boot']:.3f}, "
            f"better in {row['wins']}/{len(seeds)} repeats")
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    data = Data()
    repeats = load_run_dir(RA / "repeats")
    seeds = sorted(repeats)
    models = [m for m in MODEL_ORDER if all(m in repeats[s] for s in seeds)]
    oof = {m: [oof_predictions(repeats[s][m], data)[0] for s in seeds] for m in models}
    W = bootstrap_weights(data, N_BOOT, rng)
    say(f"Extra checks -- Run A: {len(seeds)} repeats, bootstrap B={N_BOOT}")
    R = {"cgt_per_class": cgt_per_class(data, oof, models, W),
         "ratio_sensitivity": ratio_sensitivity(repeats, models),
         "ensemble": ensemble(repeats, data, oof, models, W)}
    (OUT / "extra_checks.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
    (OUT / "extra_checks.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] wrote {OUT / 'extra_checks.txt'}")


if __name__ == "__main__":
    main()
