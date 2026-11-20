# UnifiedBench: A Cross-Dataset Benchmark for Multi-Class Smart Contract Vulnerability Classification

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
| Reentrancy (SWC-107)                     | 50                   | 30                | 17  | 47         | 97    |
| Time Manipulation (SWC-116)              | 50                   | 5                 | 1   | 6          | 56    |
| Unchecked Low-Level Calls (SWC-104)      | 100                  | 51                | 19  | 70         | 170   |
| **Total**                                | **350**              | **123**           | **120** | **243** | **593** |

- **Synthetic corpus:** [SolidiFI](https://github.com/DependableSystemsLab/SolidiFI-benchmark), 50 base contracts,
  each injected once per weakness type (`Unchecked-Send` and `Unhandled-Exceptions` both map to Unchecked Low-Level
  Calls).
- **SmartBugs Curated:** [SmartBugs Curated](https://github.com/smartbugs/smartbugs-curated), canonical category
  folders (125 candidate files). 68 of the files are contracts deployed on Ethereum (Etherscan), the others hand-written
  examples from security write-ups, SWC-registry test cases and benchmark suites.
  *Annotation Leakage Prevention:* To prevent test-set ground-truth leakage, curators' annotation comments
  (`@vulnerable_at_lines`, `// <yes> <report> CATEGORY`) and general source comments are stripped from model inputs.
  *Multi-Label Resolution:* One contract (`0x627fa62…`) originally appeared under both Reentrancy and Unchecked
  Low-Level Calls with identical code; in alignment with the single-label classification protocol and the CGT
  multi-label exclusion policy (`--cgt-multilabel drop`), this ambiguous multi-label contract pair is excluded,
  yielding 123 single-label SmartBugs contracts.
- **CGT:** the [Consolidated Ground Truth](https://github.com/gsalzer/cgt), which consolidates published ground-truth
  sets, mostly from tool evaluations (Zeus, eThor, ContractFuzzer, EthRacer, …) whose contracts their authors assessed
  manually, and maps them to SWC codes. An entry is selected when its property holds, its SWC code maps onto the six classes, it does not originate
  from SolidiFI, and its **pragma-insensitive source fingerprint** (`fp_sol2`) names a stored source file. Because
  CGT stores sources under the full fingerprint `fp_sol`, and `fp_sol2` equals `fp_sol` when a source declares no
  compiler-version pragma, this rule selects, in practice, contracts without a version pragma (none of the 120
  retained source files declares one). *Generalizability Scope:* This constitutes a deliberate representativeness
  boundary; the CGT subset reflects legacy Solidity codebases without compiler pragma constraints. CGT has one row
  per assessment (dataset entry, property), so the rows of a contract (and of every contract with the same normalised
  source) are merged into one label set; the 261 candidates that carry more than one class are excluded
  (`--cgt-multilabel drop`, the default; `--cgt-multilabel rarest` keeps them under their rarest class), and 3 exact
  duplicates of already-kept files are removed, leaving 120 single-label CGT contracts.
- **Clone families (cross-validation groups):** built with Union-Find over the whole corpus. All SolidiFI files
  derived from the same base contract share a group; files with identical normalised source share a group; any two
  contracts whose word-level token sets have Jaccard similarity >= 0.80 share a group. This gives 214 groups
  (SolidiFI 35, real-world 179); no group mixes the two sources, and no group ever spans a fold boundary (asserted).

`paper_results/manifests/` holds the two provenance manifests of this build: every SolidiFI file with its base
contract and group, and every real-world candidate with its origin, full CGT label set, SWC mapping, normalised
SHA-256 hash, group and inclusion decision. The exact upstream versions are the commits pinned in `setup_data.py`
(the `dataset_version` column of the manifest is informational).

**Note on SolidiFI:** the injected SolidiFI functions carry the weakness type in their identifiers (for example
`_re_ent`, `_tmstmp`, `_TOD`, `_intou`, `_unchk`, `bug_txorigin`). In the run profiles `runA` and `runB` every such
identifier is renamed to a neutral name (`sym1`, `sym2`, …, numbered per file) before any representation is computed,
so no identifier reveals the injected class (`train/text_utils.py`).

**Label-confidence tiers.** Every contract is assigned a provenance-based tier: Tier 1 (weakness location documented:
all SolidiFI and SmartBugs Curated contracts), Tier 2 (asserted by one or more CGT campaigns, not contradicted; 103
CGT contracts), Tier 3 (contested by another CGT campaign; 17 CGT contracts). The tier of every contract is in
`paper_results/analysis/label_confidence_tiers.csv` (written by `reviewer_analyses.py`).

## Models

All eight models use the same folds, the same sample weights and the same run seed in every repeat. Every training
contract is weighted by the balanced weight of its class times a source factor that gives the SolidiFI and the
real-world part of the training split equal total weight. In the main runs (`runA`) every model uses a single
configuration of common settings for its model family, so comparisons hold under this fixed budget rather than at
each model's own optimum; the supplementary run `runB` adds a nested hyperparameter search (`train/tuning.py`).

| Key               | Model                | Representation                                                                 | Classifier                                   |
|-------------------|----------------------|--------------------------------------------------------------------------------|----------------------------------------------|
| `xgboost`         | XGBoost (fused)      | 406-d: 384-d `all-MiniLM-L6-v2` embedding + 22 regex static features, Gaussian-noise augmentation of the training split | XGBoost, 200 trees, depth 6, lr 0.1          |
| `smartembed`      | SmartEmbed-style     | Structural token stream, skip-gram Word2Vec (d=100), mean-pooled               | Random Forest, 100 trees                     |
| `eth2vec`         | ETH2Vec-style        | `solc`-compiled runtime bytecode, disassembled opcodes, PV-DM Doc2Vec (d=100); train and test vectors both inferred | RBF SVM (C=1)                                |
| `deepsolid`       | DeepSolid            | Fused 406-d vector, Gaussian-noise augmentation of the training split          | Dual-branch MLP (Tanh semantic / ReLU static)|
| `hybridvuldetect` | HybridVulDetect      | Fused 406-d vector, Gaussian-noise augmentation of the training split          | Dual-activation MLP (parallel ReLU / Tanh)   |
| `codebert`        | CodeBERT             | Comment-free source, up to six 510-token chunks, chunk probabilities averaged  | Fine-tuned `microsoft/codebert-base`         |
| `smartbert`       | SmartBERT            | Comment-free source, up to six 510-token chunks, chunk probabilities averaged  | Fine-tuned `web3se/SmartBERT-v3`             |
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
split is augmented. DeepSolid and HybridVulDetect stop early (patience 15, at most 100 epochs) on a validation set
of about 10% of the training split that consists of whole clone families and is drawn **before** the augmentation; it
is never augmented, so it contains neither noisy copies of training contracts nor members of their clone families.
The test folds are never used for any choice.

**Leakage controls.** All comments are removed from every model input by a scanner that skips string literals
(`train/text_utils.py`), and the data loader aborts if any SmartBugs Curated annotation marker (`<yes> <report>`,
`@vulnerable_at_lines`, `@source:`, `@author:`) remains. Clone families never cross a fold boundary, and every learned
representation is fitted on the training split of a fold only.

**Input length.** CodeBERT and SmartBERT split each contract into 510-token chunks; contracts of up to six chunks
(3,060 tokens) are read in full, and for longer contracts six chunks are taken evenly spaced, always including the
first and the last. Each chunk carries 1/n of its contract's training weight, and the chunk probabilities are averaged
at test time. The `all-MiniLM-L6-v2` encoder reads 256 tokens at a time; its embedding is the mean over up to 16
evenly spaced 256-token chunks (about 4,000 tokens). 86% of the real-world contracts are read in full by the
Transformers.

**Run profiles.** The environment variable `UB_PROFILE` selects the configuration (`train/config.py`):

| Profile | Used for | Settings |
|---|---|---|
| `runA` | all main results of the paper | comment removal, chunked inputs, neutral SolidiFI identifiers, source-balanced weights |
| `runB` | Supplementary Section S6 | `runA` + nested hyperparameter search + the 261 multi-label CGT contracts (any-match scoring) + risk-ranked chunk selection for long contracts |
| `base` | development baseline | comment removal and the fixed MLP validation split only |

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
- **Per-repeat comparisons:** the pooled real-world macro-F1 of every model in each of the ten repeats (each repeat
  retrains every model on a new split with a new seed), with a sign test.
- **Training-seed study:** the five neural models are retrained with seeds 43–46 on the folds of the first repeat.
- **Leave-one-source-out:** train on all SolidiFI contracts and test on all real-world contracts, and the reverse,
  for seeds 42–46. In all LOSO runs, **no target-source contract is used during supervised fine-tuning / downstream training**,
  ensuring strict domain transfer separation while acknowledging that foundational encoders (e.g. CodeBERT, SmartBERT)
  may have encountered public smart contracts during generic unsupervised pre-training.

## Results reported in the paper

Fold-level performance (mean ± SD over 50 folds; 95% CI of macro-F1 from the corrected t-statistic):

| Model | Macro-F1 | 95% CI | Accuracy | MCC | s/fold |
|---|---:|---:|---:|---:|---:|
| SmartBERT | 0.876 ± 0.053 | [0.820, 0.933] | 0.871 | 0.841 | 118.2 |
| CodeBERT | 0.862 ± 0.055 | [0.804, 0.920] | 0.855 | 0.821 | 118.4 |
| XGBoost (fused) | 0.835 ± 0.044 | [0.789, 0.882] | 0.825 | 0.785 | 4.4 |
| GNN (Contract-GCN) | 0.830 ± 0.063 | [0.763, 0.897] | 0.825 | 0.783 | 6.4 |
| SmartEmbed | 0.798 ± 0.068 | [0.726, 0.869] | 0.785 | 0.735 | 1.9 |
| HybridVulDetect | 0.761 ± 0.078 | [0.678, 0.844] | 0.755 | 0.714 | 7.3 |
| DeepSolid | 0.751 ± 0.088 | [0.659, 0.844] | 0.747 | 0.707 | 8.1 |
| ETH2Vec | 0.681 ± 0.066 | [0.612, 0.751] | 0.665 | 0.588 | 14.4 |

Under the corrected t-test with Holm adjustment over all 28 pairs, only the five differences between ETH2Vec and the
five strongest models are significant.

Macro-F1 by source (out-of-fold predictions, mean over repeats; real-world 95% CI from the clone-group bootstrap):

| Model | All | SolidiFI | Real-world [95% CI] | SmartBugs | CGT | Real-world − SolidiFI |
|---|---:|---:|---:|---:|---:|---:|
| SmartBERT | 0.884 | 1.000 | 0.621 [0.518, 0.701] | 0.669 | 0.425 | −0.379 |
| CodeBERT | 0.870 | 1.000 | 0.572 [0.475, 0.647] | 0.625 | 0.333 | −0.428 |
| XGBoost | 0.844 | 0.979 | 0.533 [0.429, 0.621] | 0.550 | 0.337 | −0.446 |
| GNN | 0.842 | 0.988 | 0.490 [0.407, 0.559] | 0.493 | 0.345 | −0.497 |
| HybridVulDetect | 0.771 | 0.922 | 0.465 [0.379, 0.536] | 0.497 | 0.303 | −0.457 |
| DeepSolid | 0.762 | 0.906 | 0.465 [0.378, 0.538] | 0.500 | 0.309 | −0.442 |
| SmartEmbed | 0.810 | 0.947 | 0.394 [0.321, 0.456] | 0.352 | 0.311 | −0.553 |
| ETH2Vec | 0.694 | 0.853 | 0.261 [0.201, 0.310] | 0.284 | 0.179 | −0.592 |

SmartBERT has the highest real-world point estimate under every aggregation examined, but its advantage over
CodeBERT, XGBoost and the GNN is not significant under the corrected t-test.

Leave-one-source-out macro-F1 (mean over five seeds): trained on SolidiFI only, every model reaches 0.11–0.30 on
real-world contracts; trained on real-world contracts only, the models reach 0.11–0.35 on SolidiFI. The full
statistics are in `paper_results/analysis/`:

| File | Contents |
|---|---|
| `report.txt` | pairwise tests, per-source and per-class F1, compilable-subset comparison, seed study, LOSO |
| `reviewer_report.txt` | label-confidence tiers, SmartBugs Curated vs. CGT, CGT campaigns, input length, alternative aggregations, Run B |
| `extra_checks.txt` | CGT per-class intervals, test-to-training-ratio sensitivity, majority-vote ensemble |
| `contamination_report.txt` | SmartBERT masked-language-model memorisation check |

## Repository layout

```
.
├── setup_data.py                    # download the three source corpora (pinned commits) into data/
├── build_unified_dataset.py         # CGT selection, dedup, clone grouping, build the benchmark + manifests
├── gen_dataset_stats.py             # dataset and per-fold statistics
├── run_queue.py                     # runs all experiments of a profile in order (resumable)
├── progress.py                      # shows how many folds of a run are finished
├── run_all.py                       # train/evaluate the eight models, repeated grouped 5-fold CV
├── cross_source_eval.py             # leave-one-source-out (LOSO) transfer experiment, several seeds
├── analyze_results.py               # corrected repeated-CV t-tests, cluster-bootstrap CIs, LOSO summary
├── reviewer_analyses.py             # label tiers, SmartBugs vs. CGT, input length, alternative aggregations, Run B
├── extra_checks.py                  # CGT per-class CIs, ratio sensitivity of the t-test, majority-vote ensemble
├── contamination_check.py           # SmartBERT masked-language-model memorisation check
├── generate_permutation_importance.py  # permutation importance for the fused XGBoost model (grouped folds)
├── plot_paper_figures.py            # the paper's result figures
├── requirements.txt
├── paper_results/                   # archived outputs of the runs reported in the paper (see below)
└── train/
    ├── config.py                    # paths, labels, seeds, run profiles, model hyperparameters
    ├── data_loader.py               # reads the benchmark and manifests, annotation check
    ├── text_utils.py                # comment removal, annotation check, SolidiFI identifier renaming, risk-ranked chunks
    ├── tuning.py                    # nested hyperparameter search (profile runB)
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

Run the steps from the repository root, in order. Set `UB_PROFILE=runA` (the configuration of all main results) and
`PYTHONHASHSEED=0` so that the Word2Vec and Doc2Vec models are repeatable (bash: `export UB_PROFILE=runA
PYTHONHASHSEED=0`; PowerShell: `$env:UB_PROFILE = "runA"; $env:PYTHONHASHSEED = 0`). `run_queue.py` sets both itself.

```bash
# 1. Download SolidiFI, SmartBugs Curated and CGT at the pinned commits into data/
python setup_data.py

# 2. Build the benchmark (data/synthetic_benchmark/, data/expanded_real_world_set/ and their manifests)
python build_unified_dataset.py
#    and, for the supplementary Run B only, the variant that keeps the multi-label CGT contracts
python build_unified_dataset.py --cgt-multilabel keep

# 3. Dataset / per-fold statistics -> results/dataset_stats.{json,txt}
python gen_dataset_stats.py

# 4. All Run A experiments -> results/runA/: ten repeats of grouped 5-fold CV (seeds 42..51), leave-one-source-out
#    transfer (seeds 42..46), training-seed study (seeds 43..46 on the folds of the first repeat) and the grouping
#    comparison (one group per SolidiFI file)
python run_queue.py runA

# 5. Supplementary Run B (nested search + multi-label CGT contracts, three repeats) -> results/runB/
python run_queue.py runB

# 6. Statistics -> results/runA/analysis/
python analyze_results.py --results results/runA      # report.txt, analysis.json, main_table.csv, pairwise.csv
python reviewer_analyses.py                           # reviewer_report.txt, label_confidence_tiers.csv
python extra_checks.py                                # extra_checks.txt
python contamination_check.py                         # contamination_report.txt

# 7. Permutation importance of the fused XGBoost model -> results/permutation_importance*, results/plots/
python generate_permutation_importance.py

# 8. Paper figures -> results/runA/paper_figures/
python plot_paper_figures.py --results results/runA
```

Every stage is resumable: finished folds are checkpointed, so rerunning the same command continues where it stopped
(create a file named `STOP` in the repository root to pause after the current fold; `python progress.py results/runA`
shows the progress). `run_all.py` can also be called directly and accepts `--models`, `--skip`, `--repeats`, `--seeds`
and `--overwrite`. The grouping comparison reported in the paper is, per model, the mean fold macro-F1 in
`ablation_singleton_groups/seed_42/<model>` minus that in `repeats/seed_42/<model>`.

## Reproducing the paper's numbers

**From the archived outputs.** `paper_results/` holds the per-fold predictions of every run in the paper
(`repeats/`, `seed_study/`, `loso/`, `ablation_singleton_groups/`, and `runB/` for the supplementary run), the
analysis outputs (`analysis/`), the dataset statistics, the permutation-importance outputs, the two manifests and the
three figures. After steps 1–2 above (the analysis needs the built dataset to map predictions to contracts):

```bash
python analyze_results.py   --results paper_results   # rewrites paper_results/analysis/report.txt etc.
python plot_paper_figures.py --results paper_results   # writes paper_results/paper_figures/
```

`reviewer_analyses.py`, `extra_checks.py` and `contamination_check.py` read `results/runA/` (and `results/runB/`); to
run them on the archive, copy `paper_results/` to `results/runA/` and `paper_results/runB/` to `results/runB/` first.
The permutation-importance figure (`figures/fig_perm_importance.png`) is the `results/plots/perm_importance_top20.png`
plot of step 7. The archive was produced on Windows; its text files are stored with Windows line endings
(`.gitattributes`), so files regenerated on other systems can differ in line endings only.

**By retraining.** Steps 1–2 rebuild exactly the same 593 contracts, labels and clone groups (the manifests are
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
| `results/runA/repeats/seed_<s>/<model>/fold_results.json`   | `run_queue.py runA` (`run_all.py`)    |
| `results/runA/seed_study/seed_<s>/<model>/fold_results.json` | `run_queue.py runA` (seed study)      |
| `results/runA/loso/seed_<s>/<direction>/<model>/fold_results.json`, `loso_summary.json` | `run_queue.py runA` (`cross_source_eval.py`) |
| `results/runA/ablation_singleton_groups/seed_42/<model>/fold_results.json` | `run_queue.py runA` (grouping comparison) |
| `results/<run>/eth2vec/compilation_stats.json`              | `run_all.py` (ETH2Vec: number of contracts that compile) |
| `results/runB/repeats/seed_<s>/<model>/fold_results.json`   | `run_queue.py runB`                   |
| `results/runA/analysis/report.txt`, `analysis.json`, `main_table.csv`, `pairwise.csv` | `analyze_results.py --results results/runA` |
| `results/runA/analysis/reviewer_report.txt`, `label_confidence_tiers.csv`, `extra_checks.txt`, `contamination_report.txt` | `reviewer_analyses.py`, `extra_checks.py`, `contamination_check.py` |
| `results/permutation_importance*.json`, `permutation_importance_report.txt`, `results/plots/perm_importance_top20.png` | `generate_permutation_importance.py` |
| `results/runA/paper_figures/*.png`                          | `plot_paper_figures.py --results results/runA` |

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
