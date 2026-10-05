# Exploratory I+II+III corrected Transformer

Place these five files next to `run_corrected_smoke.py` in the existing project root on the second computer: `build_corrected_macro.py`, `run_corrected_macro.py`, `evaluate_macro_portfolios.py`, `summarize_macro_appendix.py`, `run_corrected_macro_11.ps1`. It uses the project's `datasets/final_dataset.csv` and existing corrected technical, fundamental, portfolio input files. Keep the separate `coverage_age` training on the first computer untouched. Outputs use distinct `datasets/corrected_macro_consensus_20260926/` and `results/corrected_I_II_III_macro_seed...` folders.

The nine historical III feature names are kept. Daily values are accepted only if at least 99% of the date's rows agree to eight decimals. Ambiguous daily values become missing (in particular the early-2010 yield observations); no security row is picked at random. The runner uses only an accepted source date strictly **before** the corrected panel date, then imputes remaining missing values and fits its macro scaler with `TrainEndpoint` rows only. The historical source vintages behind the legacy macro panel have **not** been verified. Do not call this a point-in-time macro dataset or an independent test; the specification was built after examining the original test results.

The technical and accounting files are checked against the original SHA-256 values. The train, validation and test counts must match 756423, 109943 and 150436. Same dual-encoder cross-attention architecture and training settings as corrected I+II, but the second encoder input grows from 6 to 15; therefore an I+II versus I+II+III contrast mixes input information with parameter count. Report this as an exploratory supplementary contrast, not a clean causal macro effect. Test metrics are exported only after validation selects the checkpoint.

From **PowerShell** in the existing activated Python environment, run:

```powershell
Set-Location "\\nas.ads.mwn.de\go39lop\Desktop\Archive"
.\run_corrected_macro_11.ps1
```

The launcher builds the daily panel on CPU, performs a data-only validation with no test inference, runs 11 GPU seeds sequentially and evaluates the common portfolio rule at 10 bp. If validation or any seed fails, it stops. It creates `corrected_macro_run.log` and finally `results/macro_appendix_evaluation/macro_appendix_summary.json` plus portfolios and logs. If an error occurs, send the tail of the log. Do not rerun from seed 42 after some runs complete without checking for duplicate output folders; the summary refuses to pick among duplicates.
