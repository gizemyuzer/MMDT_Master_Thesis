# Cleanup report

The uploaded archive was cleaned conservatively. Archived experiments and source implementations were retained to preserve provenance; their presence does not imply that every historical script is a corrected entry point.

## Removed files

| Path | Bytes |
| --- | ---: |
| `TUM_MMDT_Thesis_Chapters_1-4_Draft_FINAL.docx` | 43,550 |
| `apply_patch.py` | 8,384 |
| `check_nan_temp.py` | 867 |
| `models/__pycache__/losses.cpython-310.pyc` | 2,446 |
| `models/__pycache__/pytorch_trainer.cpython-310.pyc` | 12,635 |
| `models/__pycache__/transformer_model.cpython-310.pyc` | 12,282 |
| `models/__pycache__/xgboost_model.cpython-310.pyc` | 10,720 |
| `print_cols.py` | 130 |
| `print_splits.py` | 781 |
| `results/.DS_Store` | 8,196 |
| `run_xgb_factorial.py.bak` | 15,080 |
| `thesis_text_extracted.txt` | 91,093 |
| `uv.lock` | 280,415 |
| `visualization/.DS_Store` | 6,148 |

## Documentation and environment changes

- Added root `README.md` with corrected entry points, input prerequisites, commands, result selection and limitations.
- Replaced stale macro instructions with commands matching the included files; documented the incompatible consensus-summary script.
- Normalised literal escaped line breaks in `requirements.txt`; all version pins were preserved.
- Updated `pyproject.toml` core dependencies to match the supplied environment instead of its conflicting newer versions; removed its stale `uv.lock`.
- Extended `.gitignore` for generated caches, local credentials, environments and backup files.
- Kept `deneme1.py` because a plotting utility imports it.

## Explicit limitations

- The approximately 1 GB dataset was intentionally excluded from the upload by the author. Keep the original local `datasets/` folder for training; its ingestion/universe-selection modules are also absent from this snapshot.
- No scientific experiment was rerun and no recorded result was edited.
- Full reproduction, installation availability and external WRDS/SEC access were not verified.

## Verification completed

- Parsed all 100 retained Python files successfully.
- Verified SHA-256 equality for 1351 unmodified original files.
- Validated 81 requirements lines.
- Confirmed one completed archived run per seed 42–52 for all five Transformer configuration/control groups.
- Checked corrected local module references.
- ZIP CRC validation completed after packaging.
