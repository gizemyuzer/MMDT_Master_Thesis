# Transformer availability and filing age control

This experiment is an exploratory post-test contrast with the archived, repaired I+II+IV Transformer. It was specified after seeing the original test results. It is **not** an independent confirmation or an isolated causal effect of filing language.

`run_coverage_age_control.py` keeps the original 32 technical, 6 accounting, and 19 filing input slots; only `TXT_HasCoverage` and `TXT_DaysSinceFiling` vary in the 19 filing slots. The other 17 channels are set to zero **after fitting the original train-only imputer and scaler**, before constructing any train/validation/test windows. Thus dimensions, architecture, 20-session history, training/validation/test dates, horizon purging, sampler, loss, seed and validation model selection match `run_repaired_text.py`. Zero represents each channel's train-only median after the RobustScaler center operation. No old checkpoint is resumed.

Place the four files (`run_coverage_age_control.py`, `run_coverage_age_11.ps1`, `run_coverage_age_11.sh`, `evaluate_coverage_age_control.py`) next to `run_repaired_text.py` in the **existing project directory on the A40 computer**. The existing `datasets/` directory and the original 11 runs (I, I+II, repaired I+II+IV) must be present. Use the project's working Python environment.

From **PowerShell on Windows**, with the project environment activated, run:

```powershell
Set-Location "C:\path\to\your\MMDT_Master_Thesis-main"
.\run_coverage_age_11.ps1
```

If PowerShell blocks the local script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same window, then repeat the command. WSL is unnecessary. The `.sh` launcher is only for Linux.

The launcher runs the self-check, one GPU smoke run, eleven fresh sequential seeds 42–52, then the original `evaluate_portfolios.py` at its default 10 bp to make a paired comparison. Each completed seed writes a separate `results/coverage_control_train_seed...` folder with reports and NPZ predictions. The evaluation writes `results/coverage_age_evaluation/paired_portfolio_comparison.json`, two portfolio result directories and logs. Stop if smoke fails. Running the launcher twice creates duplicate complete seeds; `evaluate_coverage_age_control.py` intentionally refuses to choose between duplicates.

Prior repaired full-text seeds took about 40–43 minutes each on the earlier machine, so allow roughly eight hours plus data loading and evaluation on comparable hardware. This estimate is not a guarantee for an A40. Do not edit the thesis tables until the eleven runs and the portfolio evaluation finish. The comparison should report paired AP, ROC-AUC, MCC, Calmar and their distribution, with its exploratory status explicit.
