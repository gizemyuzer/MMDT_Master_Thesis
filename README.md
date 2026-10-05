# Fusion Forecasting: Deep Multi-Modal Models for Equity Drawdown Risk

Research code and archived experiment outputs for Gizem Yüzer's TUM master's thesis. The study compares a dual-encoder Transformer with an Optuna-tuned XGBoost baseline, using classification and a common portfolio rule.

## Scope and reproduction limits

This is a cleaned research snapshot, not a standalone data distribution. **The large `datasets/` directory was omitted from this upload because the dataset is approximately 1 GB.** This is an intentional size limit, not evidence of missing data on the author’s computer. Full training uses the original local panels, their feature allowlists and build reports; keep the local `datasets/` folder alongside these scripts. CRSP/Compustat sources require authorised WRDS access. Archived predictions, checkpoints, preprocessing objects and reports are retained in `results/`; older experiments are retained as provenance and must not be mixed with corrected runs.

The corrected partitions use 2010–2019 for training, 2020–2021 for validation and 2022–2024 for testing, with 756,423 / 109,943 / 150,436 retained endpoints. Training seeds are 42–52. Configuration I denotes technical indicators, II accounting features, III macroeconomic features and IV SEC filing features. Filing linkage remains provisional where indicated in the reports; do not interpret `PROVISIONAL_PASS` as full historical identity verification.

## Repository map

| Files | Role |
| --- | --- |
| `fetch_permno_sources.py`, `build_price_base.py`, `build_corrected_technical_panel.py` | PERMNO-based price sources, return ledger, indicators and labels |
| `fetch_fundamental_sources.py`, `build_fundamental_panel.py` | Accounting sources and aligned daily sidecar |
| `prepare_text_reuse.py`, `build_filing_feature_cache.py`, `build_daily_text_panel.py`, `apply_text_identity_fixes.py` | Filing reuse, features, daily alignment and targeted identity repairs |
| `run_corrected_smoke.py` | Corrected loader and small integration check |
| `run_corrected_technical.py`, `run_corrected_experiment.py`, `run_repaired_text.py` | Transformer I, I+II and repaired I+II+IV |
| `run_corrected_xgboost.py` | Optuna search, frozen parameters and seed runs |
| `build_corrected_macro.py`, `corrected_macro_adapter.py`, `run_corrected_macro.py` | Exploratory nine-feature macro extension |
| `run_coverage_age_control.py`, `evaluate_coverage_age_control.py` | Exploratory filing availability/age control |
| `evaluate_portfolios.py`, `evaluate_macro_portfolios.py` | Portfolio evaluation from reduced inputs and indexed predictions |
| `explain_repaired_text.py` | Grouped Kernel SHAP for a repaired Transformer checkpoint |
| `models/` | Architectures, losses and training routines |
| `results/` | Archived reports, predictions, preprocessing and checkpoints |
| `visualization/` | Plotting utilities and historical figures |

Scripts such as `run_pipeline.py`, `run_all.py`, `train.py`, `run_modality_v2.py`, `run_xgb_factorial.py`, `run_purged_*` and early portfolio experiments are **historical workflows**, not the corrected training entry points. Several require the omitted `datasets.feature_engineering` module. They remain in place to preserve existing imports and research history. `deneme1.py` is retained because `plot_drawdown.py` imports it.

## Environment

Use Python 3.10 and preferably the original experiment environment. `requirements.txt` preserves the supplied environment's exact package versions; its literal escaped line separators have been converted to valid newlines. `pyproject.toml` now uses compatible core versions; the obsolete conflicting `uv.lock` was removed. Installation availability and fresh end-to-end training have not been verified in this cleanup.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The PyTorch GPU build must match the machine's supported driver; a successful pip installation alone does not establish CUDA availability. Do not share passwords or `.pgpass` files.

## Required local inputs

Default corrected runners expect these paths, relative to the repository root:

```text
datasets/corrected_technical_20260919T155939Z_31e99bbd/technical_panel.csv
datasets/fundamental_panel_20260919T220205Z_2aed9ab0/fundamental_daily.csv
datasets/text_identity_repair_20260923T172930Z_fe9420be/text_daily.csv
```

Restore each complete containing folder: loaders also read allowlists and JSON build reports and check source hashes. Changing paths is supported with `--technical`, `--fundamental` and, where available, `--text`. Rebuilding panels creates new timestamped folders; pass the resulting paths explicitly. Do not substitute the reduced portfolio input for the full training panel.

A source rebuild follows: frozen `universe.csv` → PERMNO price exports → price/exit review → corrected technical labels → fundamental sources/sidecar → filing archive and identity audit → repaired text sidecar → smoke → training. Universe-selection code, full licensed sources, filing archives and some manually reviewed inputs are absent from this snapshot, so this is a dependency sequence, not a complete one-command rebuild.

## Corrected training

Run commands from the repository root. These create new output folders rather than resume historical checkpoints. Do not rerun training merely to inspect archived results.

```powershell
python run_corrected_smoke.py --self-test-only
python run_corrected_smoke.py --device cuda
python run_repaired_text.py --self-test-only
python run_repaired_text.py --mode smoke --seed 42 --device cuda
```

After the checks and input audits pass, the three primary Transformer configurations can be trained sequentially:

```powershell
foreach ($seed in 42..52) {
    python run_corrected_technical.py --seed $seed --device cuda
    if ($LASTEXITCODE -ne 0) { throw "I training failed" }
    python run_corrected_experiment.py --seed $seed --device cuda
    if ($LASTEXITCODE -ne 0) { throw "I+II training failed" }
    python run_repaired_text.py --mode train --seed $seed --device cuda --allow-candidate-linkage
    if ($LASTEXITCODE -ne 0) { throw "Repaired text training failed" }
}
```

The final flag explicitly accepts the documented provisional historical filing linkage; it does not resolve that limitation. Validation selects checkpoints and classification thresholds. The test sample is for evaluation, not further tuning.

```powershell
python run_corrected_xgboost.py --self-test-only
python run_corrected_xgboost.py --mode smoke --device cpu
python run_corrected_xgboost.py --mode run --device cpu --trials 30 --seeds 42 43 44 45 46 47 48 49 50 51 52
```

XGBoost receives flat feature sequences and uses validation average precision for parameter selection. See the archived `protocol.json`, `ALL_PARAMETERS_FROZEN.json`, `completed_runs.json` and per-run reports in `results/corrected_xgb_run_20260924T100253Z_8197b145/` for the actual recorded run.

## Which archived results to use

| Configuration | Folder prefix | Completed report status |
| --- | --- | --- |
| Transformer I | `corrected_I_seed` | `PASS` |
| Transformer I+II | `corrected_I_II_seed` | `PASS` |
| Repaired Transformer I+II+IV | `corrected_I_II_IV_train_two_repairs_seed` | `PROVISIONAL_PASS` |
| Macro Transformer I+II+III | `corrected_I_II_III_seed` | `PASS` |
| Availability/age control | `coverage_control_train_seed` | `PROVISIONAL_PASS` |

Match configuration, protocol, seed, successful completion and `test_evaluated`; do not choose a folder solely by timestamp. Candidate text runs before the two repairs, smoke/pilot runs, failed runs and the earlier `purged_*` experiments are not interchangeable with these completed runs.

## Portfolio analysis

The included `portfolio_inputs_20260923T110426840595Z.zip` contains `technical_panel_subset.csv`, `return_ledger_subset.csv` and an export summary. It is a reduced analysis input, not full source data. The evaluator uses the same 20-session rebalance schedule, excludes the highest-scored 20%, equally weights retained names and charges 10 basis points per traded amount. Held position values drift between rebalances; documented exits settle to cash. It refuses unresolved held returns.

`evaluate_portfolios.py` requires a separate prediction ZIP with exactly:

```text
npz/I/42/val_predictions.npz
npz/I/42/test_predictions.npz
npz/I+II/42/val_predictions.npz
npz/I+II/42/test_predictions.npz
npz/I+II+IV/42/val_predictions.npz
npz/I+II+IV/42/test_predictions.npz
... the same structure for seeds 43–52
```

Copy the corresponding archived files into this layout and zip the `npz/` folder. Then:

```powershell
python evaluate_portfolios.py portfolio_inputs_20260923T110426840595Z.zip matched_transformer_predictions.zip results/portfolio_reproduction
```

Outputs are `portfolio_results.json` and `equity_curves.json`. The original evaluator is specifically for the three Transformer configuration labels; it does not automatically include the archived XGBoost runs. Calendar comparisons must slice the continuous wealth series, not restart each strategy annually.

## Supplementary analyses and known snapshot mismatch

See `COVERAGE_CONTROL_README.md` for the filing availability/age contrast. It is exploratory and creates duplicate run folders if rerun. `CORRECTED_MACRO_README.md` documents the macro code present here.

**The supplied `summarize_macro_appendix.py` expects a different consensus-macro protocol and folder prefix from the completed macro runs in this archive.** It cannot discover these runs unchanged. Its source is retained as historical evidence; do not present it as a working reproduction command. The macro training files present here instead use `build_corrected_macro.py` and `corrected_macro_adapter.py`, with explicit `--macro` support in the runner. This cleanup does not reconcile alternative macro protocols or change scientific results.

```powershell
python explain_repaired_text.py --device cuda --samples 64 --background 16 --nsamples 256
```

SHAP requires the original panels and the default repaired seed-42 checkpoint. It describes sampled model attribution, not causality or all-seed feature importance.

## Cleanup and verification

See `CLEANUP_REPORT.md` for every deletion and metadata change. Scientific Python source, experiment reports, predictions, checkpoints and figures were preserved byte-for-byte except for the explicitly removed one-off helpers. Verification covers syntax, retained-file hashes, requirements formatting, corrected local imports and ZIP integrity. No network data fetch, GPU training, pickle/checkpoint loading or new test-set evaluation was performed.
