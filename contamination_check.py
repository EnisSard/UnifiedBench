"""
contamination_check.py
----------------------
Empirical memorisation check for SmartBERT (web3se/SmartBERT-v3), whose pre-training
corpus of Solidity code cannot be inspected and may contain the public SmartBugs and
CGT contracts.

For every contract, 15% of the tokens of up to 6 evenly spaced 510-token chunks are
masked (fixed random seed) and SmartBERT's masked-LM loss on them is measured. A
contract seen during pre-training is expected to have a markedly lower loss
(memorisation). We then ask whether SmartBERT's real-world classification advantage
over CodeBERT (which was not pre-trained on Solidity) concentrates on the
low-loss, i.e. most familiar, contracts. If the advantage came from memorised
contracts, it should be largest in the lowest-loss quartile and vanish in the others.

    UB_PROFILE=runA python contamination_check.py
Writes results/runA/analysis/contamination_report.txt and contamination.json.
"""
import json
import os

import numpy as np
import torch

os.environ.setdefault("UB_PROFILE", "runA")
from analyze_results import Data, load_run_dir, oof_predictions
from train.data_loader import load_records
from train.features.fused_features import pick_evenly
from pathlib import Path

OUT = Path(__file__).parent / "results" / "runA" / "analysis"


def mlm_losses(texts, ckpt="web3se/SmartBERT-v3", mask_frac=0.15, max_chunks=6, seed=0):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForMaskedLM.from_pretrained(ckpt).to(dev).eval()
    rng = np.random.default_rng(seed)
    special = set(tok.all_special_ids)
    out = []
    for t in texts:
        ids = tok(t, add_special_tokens=False, verbose=False)["input_ids"]
        chunks = pick_evenly([ids[j:j + 510] for j in range(0, max(len(ids), 1), 510)] or [[]], max_chunks)
        tot, n = 0.0, 0
        for c in chunks:
            if len(c) < 10:
                continue
            row = torch.tensor([[tok.cls_token_id] + c + [tok.sep_token_id]])
            cand = [i + 1 for i, x in enumerate(c) if x not in special]
            k = max(1, int(round(mask_frac * len(cand))))
            pos = rng.choice(cand, k, replace=False)
            labels = torch.full_like(row, -100)
            labels[0, pos] = row[0, pos]
            inp = row.clone()
            inp[0, pos] = tok.mask_token_id
            with torch.no_grad():
                loss = model(input_ids=inp.to(dev), labels=labels.to(dev)).loss.item()
            tot += loss * k
            n += k
        out.append(tot / n if n else np.nan)
    return np.array(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    texts, labels, groups, sources, names = load_records()
    data = Data()
    loss = mlm_losses(texts)
    rep = load_run_dir(Path(__file__).parent / "results" / "runA" / "repeats")
    seeds = sorted(rep)
    acc = {m: np.mean([(oof_predictions(rep[s][m], data)[0] == data.y) for s in seeds], axis=0)
           for m in ("smartbert", "codebert")}
    lines = ["SmartBERT masked-LM loss (nats/token, 15% masked, up to 6 chunks) and classification accuracy",
             f"{'subset':<12}{'n':>5}{'median loss':>13}{'IQR':>18}{'share < 0.3':>13}"]
    R = {"loss": {}}
    for s in ("solidifi", "smartbugs", "cgt"):
        mk = data.sub == s
        v = loss[mk]
        R["loss"][s] = dict(n=int(mk.sum()), median=float(np.nanmedian(v)), q1=float(np.nanpercentile(v, 25)),
                            q3=float(np.nanpercentile(v, 75)), share_below_0_3=float(np.nanmean(v < 0.3)))
        lines.append(f"{s:<12}{int(mk.sum()):>5}{np.nanmedian(v):>13.3f}"
                     f"{f'{np.nanpercentile(v, 25):.3f}-{np.nanpercentile(v, 75):.3f}':>18}{np.nanmean(v < 0.3):>13.2f}")
    rw = data.masks["realworld"] & ~np.isnan(loss)
    q = np.nanpercentile(loss[rw], [25, 50, 75])
    bins = np.digitize(loss, q)
    lines += ["", "Real-world contracts by SmartBERT loss quartile (Q1 = most familiar); accuracy mean over 10 repeats",
              f"{'quartile':<10}{'n':>5}{'loss range':>18}{'SmartBERT acc':>15}{'CodeBERT acc':>14}{'advantage':>11}"]
    R["quartiles"] = []
    for b in range(4):
        mk = rw & (bins == b)
        a, c = float(acc["smartbert"][mk].mean()), float(acc["codebert"][mk].mean())
        R["quartiles"].append(dict(q=b + 1, n=int(mk.sum()), lo=float(loss[mk].min()), hi=float(loss[mk].max()),
                                   smartbert_acc=a, codebert_acc=c, advantage=a - c))
        lines.append(f"Q{b + 1:<9}{int(mk.sum()):>5}{f'{loss[mk].min():.2f}-{loss[mk].max():.2f}':>18}{a:>15.3f}{c:>14.3f}{a - c:>+11.3f}")
    from scipy import stats
    rho = stats.spearmanr(loss[rw], acc["smartbert"][rw] - acc["codebert"][rw])
    R["spearman_loss_vs_advantage"] = [float(rho[0]), float(rho[1])]
    lines.append(f"Spearman(loss, SmartBERT-CodeBERT accuracy advantage) over real-world contracts: rho={rho[0]:+.3f}, p={rho[1]:.3f}")
    (OUT / "contamination.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    (OUT / "contamination_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
