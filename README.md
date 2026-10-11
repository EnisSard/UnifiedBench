# UnifiedBench

Code and benchmark-construction pipeline for the paper:

> **UnifiedBench: A Cross-Dataset Benchmark for Multi-Class Smart Contract Vulnerability Classification**
> Mesut Ozdag, Enishan Sardogan, Necmeddin Gazel, Ali Karahaliloz, Crispen Chachengwa, Carlos Luengo,
> Ismail Rakip Karas, Fatih Kirisik
> Karabuk University; University of Central Florida

UnifiedBench is a clone-controlled benchmark for classifying Solidity smart contracts into six vulnerability
categories. It keeps a synthetic, bug-injection corpus strictly separate from real-world (human-written) contracts,
groups every
clone family (including all variants of the same SolidiFI base contract) before splitting, and evaluates eight
model families under one shared protocol: ten repeats of Stratified Grouped 5-fold cross-validation, corrected
resampled t-tests, clone-group bootstrap confidence intervals, a training-seed study and a strict
leave-one-source-out (LOSO) transfer experiment.

This repository contains the dataset-construction pipeline, the eight model implementations, the evaluation,
statistics and plotting scripts, and the archived outputs of the runs reported in the paper (`paper_results/`). It
does **not** bundle the third-party source datasets; `setup_data.py` downloads them at pinned versions.

## Dataset

The task is **vulnerability-type classification** among contracts already known to be flawed: each contract is
assigned exactly one of six DASP/SWC categories. There is no benign class.

| Class (DASP / SWC)                       | SolidiFI (synthetic) | SmartBugs Curated | CGT | Real-world | Total |
|------------------------------------------|---------------------:|------------------:|----:|-----------:|------:|
| Access Control (SWC-105/106/108/112/115) | 50                   | 18                | 3   | 21         | 71    |
| Arithmetic (SWC-101)                     | 50                   | 15                | 64  | 79         | 129   |
| Front-Running (SWC-114)                  | 50                   | 4                 | 16  | 20         | 70    |
| Reentrancy (SWC-107)                     | 50                   | 31                | 17  | 48         | 98    |
| Time Manipulation (SWC-116)              | 50                   | 5                 | 1   | 6          | 56    |
| Unchecked Low-Level Calls (SWC-104)      | 100                  | 52                | 19  | 71         | 171   |
| **Total**                                | **350**              | **125**           | **120** | **245** | **595** |

- **Synthetic corpus:** [SolidiFI](https://github.com/DependableSystemsLab/SolidiFI-benchmark), 50 base contracts,
  each injected once per weakness type (`Unchecked-Send` and `Unhandled-Exceptions` both map to Unchecked Low-Level
  Calls).
- **SmartBugs Curated:** [SmartBugs Curated](https://github.com/smartbugs/smartbugs-curated), every file of the six
  mapped category folders (125 files); the class is the folder category. 68 of the 125 files are contracts deployed
  on Ethereum (Etherscan), the others hand-written examples from security write-ups, SWC-registry test cases and
  benchmark suites. The files keep the curators' annotation comments (`@vulnerable_at_lines`,
  `// <yes> <report> CATEGORY`), which name the file's category; comments are not removed for the Transformer,
  MiniLM and GCN inputs. One contract (`0x627fa62…`) is filed under both Reentrancy and Unchecked Low-Level Calls;
  both files are kept with their folder labels and, having identical code (they differ only in their annotation
  comments), share a clone family.
- **CGT:** the [Consolidated Ground Truth](https://github.com/gsalzer/cgt), which consolidates published ground-truth
  sets, mostly from tool evaluations (Zeus, eThor, ContractFuzzer, EthRacer, …) whose contracts their authors assessed
  manually, and maps them to SWC codes. An entry is selected when its property holds, its SWC code maps onto the six classes, it does not originate
  from SolidiFI, and its **pragma-insensitive source fingerprint** (`fp_sol2`) names a stored source file. Because
  CGT stores sources under the full fingerprint `fp_sol`, and `fp_sol2` equals `fp_sol` when a source declares no
  compiler-version pragma, this rule selects, in practice, contracts without a version pragma (none of the 120
  retained source files declares one). CGT has one row per assessment (dataset entry, property), so the rows of a
  contract (and of every contract with the same normalised source) are merged into one label set; the 261 candidates that carry more than one class are
  excluded (`--cgt-multilabel drop`, the default; `--cgt-multilabel rarest` keeps them under their rarest class), and
  3 exact duplicates of already-kept files are removed, leaving 120 single-label CGT contracts.
- **Clone families (cross-validation groups):** built with Union-Find over the whole corpus. All SolidiFI files
  derived from the same base contract share a group; files with identical normalised source share a group; any two
  contracts whose word-level token sets have Jaccard similarity >= 0.80 share a group. This gives 214 groups
  (SolidiFI 35, real-world 179); no group mixes the two sources, and no group ever spans a fold boundary (asserted).

`paper_results/manifests/` holds the two provenance manifests of this build: every SolidiFI file with its base
contract and group, and every real-world candidate with its origin, full CGT label set, SWC mapping, normalised
SHA-256 hash, group and inclusion decision. The exact upstream versions are the commits pinned in `setup_data.py`
(the `dataset_version` column of the manifest is informational).

**Note on SolidiFI:** the injected SolidiFI functions carry the weakness type in their identifiers (for example
`_re_ent`, `_tmstmp`, `_TOD`, `_intou`, `_unchk`, `bug_txorigin`). Models that read identifiers can exploit this,
which is one reason synthetic scores are optimistic.

## Models

All eight models use the same folds, the same class-weighted loss and the same run seed in every repeat. No
hyperparameter search is run: every model uses a single configuration of common settings for its model family, so
comparisons hold under this fixed, untuned budget rather than at each model's own optimum.

| Key               | Model                | Representation                                                                 | Classifier                                   |
|-------------------|----------------------|--------------------------------------------------------------------------------|----------------------------------------------|
| `xgboost`         | XGBoost (fused)      | 406-d: 384-d `all-MiniLM-L6-v2` embedding + 22 regex static features, Gaussian-noise augmentation of the training split | XGBoost, 200 trees, depth 6, lr 0.1          |
| `smartembed`      | SmartEmbed-style     | Structural token stream, skip-gram Word2Vec (d=100), mean-pooled               | Random Forest, 100 trees                     |
| `eth2vec`         | ETH2Vec-style        | `solc`-compiled runtime bytecode, disassembled opcodes, PV-DM Doc2Vec (d=100); train and test vectors both inferred | RBF SVM (C=1)                                |
| `deepsolid`       | DeepSolid            | Fused 406-d vector, Gaussian-noise augmentation of the training split          | Dual-branch MLP (Tanh semantic / ReLU static)|
| `hybridvuldetect` | HybridVulDetect      | Fused 406-d vector, Gaussian-noise augmentation of the training split          | Dual-activation MLP (parallel ReLU / Tanh)   |
| `codebert`        | CodeBERT             | Raw source, 512 tokens                                                         | Fine-tuned `microsoft/codebert-base`         |
| `smartbert`       | SmartBERT            | Raw source, 512 tokens                                                         | Fine-tuned `web3se/SmartBERT-v3`             |
| `gnn`             | Contract-GCN         | Function/block graph, MiniLM node embeddings                                   | 2 x `GCNConv`, mean+max pooling, MLP head    |

Representations learned from data (Word2Vec, Doc2Vec, Transformer fine-tuning) are fitted inside each fold on the
training split only. ETH2Vec needs bytecode: it is trained only on contracts that compile with the `solc` version
selected for them (the closest to the declared pragma from the matrix 0.4.11 to 0.8.20; 0.4.24 when a source declares
no pragma). Non-compiling test contracts are still
scored with the fold's majority training class, so ETH2Vec is evaluated on exactly the same contracts as every
other model; `analyze_results.py` also compares all models on the compilable subset alone.

**Augmentation of the fused-feature models.** For XGBoost, DeepSolid and HybridVulDetect, every training instance of
five of the six classes (all except Unchecked Low-Level Calls) is replicated three times with additive Gaussian
noise (sigma = 0.01); see `apply_gaussian_augmentation` in `train/features/fused_features.py`. Only the training
split is augmented. DeepSolid and HybridVulDetect stop early on a 10% slice of the (augmented) training split
(patience 15, at most 100 epochs); the test folds are never used for any choice.

**Input limits.** The `all-MiniLM-L6-v2` encoder used by the fused vector and the GCN reads at most 256 tokens of
its input; Transformer inputs are truncated to 512 tokens. All models except SmartEmbed (which strips comments) and
ETH2Vec (which reads bytecode) read the source including its comments.

## Evaluation protocol

- **Cross-validation:** ten repeats of `StratifiedGroupKFold` (5 folds). Repeat r uses seed 42 + r for the split and
  for every stochastic component of every model, so each stochastic model is trained with ten seeds. All models of a
  repeat share the folds.
- **Fold-level results:** macro-F1 (primary), accuracy and Matthews correlation coefficient over the 50 fold
  estimates, with 95% confidence intervals from the corrected resampled t-statistic (Nadeau & Bengio 2003;
  Bouckaert & Frank 2004). Pairwise model comparisons use the same corrected test on paired per-fold differences,
  Holm-adjusted over the 28 pairs.
- **Per-source results** (SolidiFI, real-world, SmartBugs Curated, CGT): each repeat predicts every contract once;
  the out-of-fold predictions are pooled per repeat and averaged over repeats. 95% CIs and paired model differences
  come from a clone-group bootstrap (5,000 resamples, stratified by source), Holm-adjusted per subset. This bootstrap
  reflects the sampling of test contracts; the corrected t-test also reflects training-fold variability.
- **Training-seed study:** the five neural models are retrained with seeds 43–46 on the folds of the first repeat.
- **Leave-one-source-out:** train on all SolidiFI contracts and test on all real-world contracts, and the reverse,
  for seeds 42–46.

## Results reported in the paper

Fold-level performance (mean ± SD over 50 folds; 95% CI of macro-F1 from the corrected t-statistic):

| Model | Macro-F1 | 95% CI | Accuracy | MCC | s/fold |
|---|---:|---:|---:|---:|---:|
| GNN (Contract-GCN) | 0.842 ± 0.064 | [0.774, 0.909] | 0.843 | 0.805 | 7.8 |
| XGBoost (fused) | 0.836 ± 0.061 | [0.772, 0.900] | 0.826 | 0.785 | 6.9 |
| SmartBERT | 0.798 ± 0.087 | [0.706, 0.889] | 0.796 | 0.748 | 32.4 |
| SmartEmbed | 0.788 ± 0.077 | [0.707, 0.870] | 0.777 | 0.725 | 3.0 |
| HybridVulDetect | 0.715 ± 0.073 | [0.638, 0.792] | 0.707 | 0.651 | 17.3 |
| CodeBERT | 0.708 ± 0.094 | [0.609, 0.807] | 0.709 | 0.648 | 32.6 |
| DeepSolid | 0.680 ± 0.085 | [0.590, 0.771] | 0.672 | 0.609 | 18.7 |
| ETH2Vec | 0.660 ± 0.116 | [0.537, 0.782] | 0.645 | 0.559 | 26.9 |

Macro-F1 by source (out-of-fold predictions, mean over repeats; real-world 95% CI from the clone-group bootstrap):

| Model | All | SolidiFI | Real-world [95% CI] | SmartBugs | CGT | Real-world − SolidiFI |
|---|---:|---:|---:|---:|---:|---:|
| SmartBERT | 0.800 | 0.841 | 0.705 [0.621, 0.759] | 0.808 | 0.325 | −0.136 |
| XGBoost | 0.846 | 0.939 | 0.615 [0.518, 0.691] | 0.696 | 0.332 | −0.325 |
| DeepSolid | 0.679 | 0.675 | 0.597 [0.491, 0.676] | 0.676 | 0.309 | −0.078 |
| HybridVulDetect | 0.718 | 0.749 | 0.587 [0.485, 0.661] | 0.665 | 0.301 | −0.163 |
| GNN | 0.855 | 0.991 | 0.533 [0.443, 0.605] | 0.566 | 0.342 | −0.458 |
| CodeBERT | 0.706 | 0.772 | 0.509 [0.438, 0.560] | 0.530 | 0.273 | −0.263 |
| SmartEmbed | 0.805 | 0.946 | 0.387 [0.317, 0.447] | 0.344 | 0.314 | −0.559 |
| ETH2Vec | 0.690 | 0.850 | 0.256 [0.198, 0.307] | 0.252 | 0.185 | −0.594 |

Leave-one-source-out macro-F1 (mean over five seeds): trained on SolidiFI only, every model reaches 0.11–0.30 on
real-world contracts; trained on real-world contracts only, the GCN reaches 0.80 on SolidiFI and the other models
0.09–0.33. The full statistics (pairwise tests, per-class F1, matched ETH2Vec comparison, seed study, LOSO) are in
`paper_results/analysis/report.txt`.

## Repository layout

```
.
├── setup_data.py                    # download the three source corpora (pinned commits) into data/
├── build_unified_dataset.py         # CGT selection, dedup, clone grouping, build the benchmark + manifests
├── gen_dataset_stats.py             # dataset and per-fold statistics
├── run_all.py                       # train/evaluate the eight models, repeated grouped 5-fold CV
├── cross_source_eval.py             # leave-one-source-out (LOSO) transfer experiment, several seeds
├── analyze_results.py               # corrected repeated-CV t-tests, cluster-bootstrap CIs, LOSO summary
├── generate_permutation_importance.py  # permutation importance for the fused XGBoost model (grouped folds)
├── plot_paper_figures.py            # the paper's result figures
├── requirements.txt
├── paper_results/                   # archived outputs of the runs reported in the paper (see below)
└── train/
    ├── config.py                    # paths, labels, seeds, model hyperparameters
    ├── data_loader.py               # reads the benchmark and manifests
    ├── evaluate.py                  # StratifiedGroupKFold splits, metrics, result persistence
    ├── features/                    # fused, AST/token and opcode feature extractors
    └── models/                      # the eight model implementations
```

## Getting started

### Requirements

- Python 3.10 or newer, and the `git` command-line client (used by `setup_data.py`).
- A CUDA GPU is strongly recommended for the two Transformer models and the GNN; the reported timings were measured
  on a single RTX 4070 Ti. Everything also runs on CPU, only much more slowly.
- An internet connection on the first run: source corpora are downloaded from GitHub, model checkpoints
  (`microsoft/codebert-base`, `web3se/SmartBERT-v3`, `all-MiniLM-L6-v2`) from the Hugging Face Hub, and `py-solc-x`
  downloads the `solc` compilers used by ETH2Vec.

### Install

```bash
pip install -r requirements.txt
```

Install a `torch` build that matches your CUDA version first if you want GPU training, and make sure
`torch-geometric` matches your installed `torch` (see the notes in `requirements.txt`).

### Run the pipeline

Run the steps from the repository root, in order, with `PYTHONHASHSEED=0` set so that the Word2Vec and Doc2Vec models
are repeatable (bash: `export PYTHONHASHSEED=0`; PowerShell: `$env:PYTHONHASHSEED = 0`; cmd: `set PYTHONHASHSEED=0`).

```bash
# 1. Download SolidiFI, SmartBugs Curated and CGT at the pinned commits into data/
python setup_data.py

# 2. Build the benchmark (data/synthetic_benchmark/, data/expanded_real_world_set/ and their manifests)
python build_unified_dataset.py

# 3. Dataset / per-fold statistics -> results/dataset_stats.{json,txt}
python gen_dataset_stats.py

# 4. Ten repeats of grouped 5-fold CV (seeds 42..51), all eight models -> results/repeats/
python run_all.py

# 5. Training-seed study: folds of the first repeat, seeds 43..46 -> results/seed_study/
python run_all.py --seeds 43 44 45 46 --split-seed 42 --out results/seed_study \
       --models deepsolid hybridvuldetect codebert smartbert gnn

# 6. Leave-one-source-out transfer, seeds 42..46 -> results/loso/
python cross_source_eval.py

# 7. Permutation importance of the fused XGBoost model -> results/permutation_importance*
python generate_permutation_importance.py

# 8. Grouping comparison: first repeat with one group per SolidiFI file -> results/ablation_singleton_groups/
python run_all.py --seeds 42 --file-level-solidifi-groups --out results/ablation_singleton_groups

# 9. Statistics -> results/analysis/report.txt (+ analysis.json, main_table.csv, pairwise.csv)
python analyze_results.py

# 10. Paper figures -> results/paper_figures/
python plot_paper_figures.py
```

`run_all.py` skips (seed, model) pairs that already have results, so an interrupted run can simply be restarted
(`--overwrite` forces a re-run). It also accepts `--models`, `--skip`, `--repeats` and `--seeds`. The grouping
comparison reported in the paper is, per model, the mean fold macro-F1 in `ablation_singleton_groups/seed_42/<model>`
minus that in `repeats/seed_42/<model>`.

## Reproducing the paper's numbers

**From the archived outputs.** `paper_results/` holds the per-fold predictions of every run in the paper
(`repeats/`, `seed_study/`, `loso/`, `ablation_singleton_groups/`), the analysis outputs (`analysis/`), the dataset
statistics, the permutation-importance outputs, the two manifests and the three figures. After steps 1–2 above
(the analysis needs the built dataset to map predictions to contracts):

```bash
python analyze_results.py   --results paper_results   # rewrites paper_results/analysis/
python plot_paper_figures.py --results paper_results   # writes paper_results/paper_figures/
```

The first command reproduces the archived `analysis/` files; the second reproduces `fig_model_comparison.png` and
`fig_cm_smartbert_combined.png` of `figures/` (byte-identical with matplotlib 3.10.1, which stores its version in
the PNG). The permutation-importance figure (`figures/fig_perm_importance.png`) is the
`results/plots/perm_importance_top20.png` plot of step 7. The archive
was produced on Windows; its text files are stored with Windows line endings (`.gitattributes`), so files
regenerated on other systems can differ in line endings only.

**By retraining.** Steps 1–2 rebuild exactly the same 595 contracts, labels and clone groups (the manifests are
identical). Retraining gives numbers close to, but not bit-identical with, the archived ones: GPU kernels for the
Transformers and the GCN are not fully deterministic, results can differ across library versions and hardware, and
the archived runs used copies of the SolidiFI and SmartBugs files with Windows (CRLF) line endings, whereas
`setup_data.py` downloads the files with the line endings stored in their repositories. This affects only the raw
text the models read, not the dataset composition. The measured seed-to-seed variation is reported in the paper.

The archived runs used Python 3.10 with torch 2.7.1 (CUDA 11.8), transformers 5.5.0, sentence-transformers 5.5.0,
gensim 4.4.0, xgboost 3.2.0, scikit-learn 1.7.2, numpy 2.2.5, scipy 1.15.2, torch-geometric 2.7.0, matplotlib
3.10.1, py-solc-x 2.0.5 and pyevmasm 0.2.3, and the Hugging Face checkpoints `web3se/SmartBERT-v3` (revision
`2f9f8f16`), `microsoft/codebert-base` (`3b0952fe`) and `sentence-transformers/all-MiniLM-L6-v2` (`1110a243`).

## Outputs

Scripts write to `results/` (created on demand and excluded from version control):

| Path                                                        | Produced by                           |
|-------------------------------------------------------------|---------------------------------------|
| `results/dataset_stats.json`, `dataset_stats.txt`           | `gen_dataset_stats.py`                |
| `results/repeats/seed_<s>/<model>/fold_results.json`        | `run_all.py`                          |
| `results/seed_study/seed_<s>/<model>/fold_results.json`     | step 5 (`run_all.py --seeds 43 44 45 46 --split-seed 42 --out results/seed_study ...`) |
| `results/loso/seed_<s>/<direction>/<model>/fold_results.json`, `results/loso/loso_summary.json` | `cross_source_eval.py` |
| `results/ablation_singleton_groups/seed_42/<model>/fold_results.json` | step 8 (`run_all.py --seeds 42 --file-level-solidifi-groups --out results/ablation_singleton_groups`) |
| `results/<run>/eth2vec/compilation_stats.json`              | `run_all.py` (ETH2Vec: number of contracts that compile) |
| `results/analysis/report.txt`, `analysis.json`, `main_table.csv`, `pairwise.csv` | `analyze_results.py` |
| `results/permutation_importance*.json`, `permutation_importance_report.txt`, `results/plots/perm_importance_top20.png` | `generate_permutation_importance.py` |
| `results/paper_figures/*.png`                               | `plot_paper_figures.py`               |

## Data and licensing

The code in this repository (everything except the third-party corpora and checkpoints listed below) is released
under the [MIT License](LICENSE).

This repository does not redistribute third-party contracts. `setup_data.py` downloads SolidiFI, SmartBugs Curated
and CGT directly from their upstream repositories, and each corpus remains under its own license. Note in
particular that the CGT Solidity sources retain the license they were published under on Etherscan. Please check
the upstream licenses before redistributing any built dataset.

## Citation

If you use UnifiedBench, please cite the paper:

<!-- Replace with the journal entry (journal, volume, DOI) once the paper is published. -->
```bibtex
@misc{unifiedbench,
  title  = {UnifiedBench: A Cross-Dataset Benchmark for Multi-Class Smart Contract Vulnerability Classification},
  author = {Ozdag, Mesut and Sardogan, Enishan and Gazel, Necmeddin and Karahaliloz, Ali and Chachengwa, Crispen and Luengo, Carlos and Karas, Ismail Rakip and Kirisik, Fatih},
  year   = {2026},
  note   = {Manuscript submitted for publication}
}
```

## Acknowledgements

We thank the maintainers of [SolidiFI](https://github.com/DependableSystemsLab/SolidiFI-benchmark),
[SmartBugs Curated](https://github.com/smartbugs/smartbugs-curated) and the
[Consolidated Ground Truth](https://github.com/gsalzer/cgt) for making their data publicly available, and the
developers of [`web3se/SmartBERT-v3`](https://huggingface.co/web3se/SmartBERT-v3) for releasing their checkpoint.
