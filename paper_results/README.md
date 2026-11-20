# paper_results

Archived outputs of the runs reported in the paper. All main results come from the run profile `runA`; the
supplementary nested search and multi-label evaluation (Supplementary Section S6) come from `runB`. The tables,
statistics and result figures of the paper are computed from these files (the token-length statistics and the
constant-predictor baseline are computed from the built dataset itself).

| Path | Contents |
|---|---|
| `repeats/seed_<42..51>/<model>/fold_results.json` | Ten repeats of Stratified Grouped 5-fold CV: per-fold metrics, `y_true` / `y_pred` / sources |
| `*/eth2vec/compilation_stats.json` | Number of contracts that compile to bytecode (ETH2Vec), per run |
| `seed_study/seed_<43..46>/<model>/fold_results.json` | Training-seed study (folds of the first repeat; seed 42 is `repeats/seed_42`) |
| `loso/seed_<42..46>/<direction>/<model>/fold_results.json`, `loso/loso_summary.json` | Leave-one-source-out runs |
| `ablation_singleton_groups/seed_42/<model>/fold_results.json` | First repeat with one group per SolidiFI file (grouping comparison) |
| `runB/repeats/seed_<42..44>/<model>/fold_results.json`, `runB/analysis/rows.json` | Supplementary run: nested search and multi-label CGT contracts |
| `analysis/report.txt`, `analysis.json`, `main_table.csv`, `pairwise.csv` | Main statistics (output of `analyze_results.py`) |
| `analysis/reviewer_report.txt`, `reviewer_analysis.json` | Label-confidence tiers, SmartBugs Curated vs. CGT, CGT campaigns, input length, alternative aggregations, Run B summary (`reviewer_analyses.py`) |
| `analysis/label_confidence_tiers.csv` | Label-confidence tier of every contract |
| `analysis/extra_checks.txt`, `extra_checks.json` | CGT per-class intervals, test-to-training-ratio sensitivity, majority-vote ensemble (`extra_checks.py`) |
| `analysis/contamination_report.txt`, `contamination.json` | SmartBERT memorisation check (`contamination_check.py`) |
| `dataset_stats.json`, `dataset_stats.txt` | Dataset composition and per-fold class counts |
| `permutation_importance*.json`, `permutation_importance_report.txt` | Permutation importance of the fused XGBoost model |
| `manifests/synthetic_manifest.csv`, `manifests/reproducibility_manifest.csv` | Provenance of every SolidiFI file and every real-world candidate |
| `figures/*.png` | The three result figures of the paper |

Regenerate the main statistics and figures from these files (after `setup_data.py` and `build_unified_dataset.py`,
which the analysis needs to map predictions to contracts):

```bash
python analyze_results.py    --results paper_results   # rewrites analysis/report.txt, analysis.json, *.csv
python plot_paper_figures.py --results paper_results   # writes paper_figures/
```

`figures/fig_perm_importance.png` comes from `generate_permutation_importance.py`. The additional analyses
(`reviewer_analyses.py`, `extra_checks.py`, `contamination_check.py`) read `results/runA/` and `results/runB/`; copy
this folder to `results/runA/` and `runB/` to `results/runB/` to rerun them on the archive. The files were produced on
Windows and are stored with Windows line endings, so regenerating them elsewhere can differ in line endings only.
