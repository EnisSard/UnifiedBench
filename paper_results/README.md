# paper_results

Archived outputs of the runs reported in the paper. The tables, statistics and result figures of the paper are
computed from these files (the token-length statistics and the constant-predictor baseline are computed from the
built dataset itself).

| Path | Contents |
|---|---|
| `repeats/seed_<42..51>/<model>/fold_results.json` | Ten repeats of Stratified Grouped 5-fold CV: per-fold metrics, test indices, `y_true` / `y_pred` / sources |
| `*/eth2vec/compilation_stats.json` | Number of contracts that compile to bytecode (ETH2Vec), per run |
| `seed_study/seed_<43..46>/<model>/fold_results.json` | Training-seed study (folds of the first repeat; seed 42 is `repeats/seed_42`) |
| `loso/seed_<42..46>/<direction>/<model>/fold_results.json`, `loso/loso_summary.json` | Leave-one-source-out runs |
| `ablation_singleton_groups/seed_42/<model>/fold_results.json` | First repeat with one group per SolidiFI file (grouping comparison) |
| `analysis/report.txt`, `analysis.json`, `main_table.csv`, `pairwise.csv` | All statistics (output of `analyze_results.py`) |
| `dataset_stats.json`, `dataset_stats.txt` | Dataset composition and per-fold class counts |
| `permutation_importance*.json`, `permutation_importance_report.txt` | Permutation importance of the fused XGBoost model |
| `manifests/synthetic_manifest.csv`, `manifests/reproducibility_manifest.csv` | Provenance of every SolidiFI file and every real-world candidate |
| `figures/*.png` | The three result figures of the paper |

Regenerate the statistics and figures from these files (after `setup_data.py` and `build_unified_dataset.py`, which
the analysis needs to map predictions to contracts):

```bash
python analyze_results.py    --results paper_results   # rewrites analysis/
python plot_paper_figures.py --results paper_results   # writes paper_figures/
```

The first command reproduces `analysis/`; the second reproduces `figures/fig_model_comparison.png` and
`figures/fig_cm_smartbert_combined.png` (byte-identical with matplotlib 3.10.1). `figures/fig_perm_importance.png`
comes from `generate_permutation_importance.py`. The files were produced on Windows and are stored with Windows
line endings, so regenerating them elsewhere can differ in line endings only.
