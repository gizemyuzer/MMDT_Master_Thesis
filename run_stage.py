"""
run_stage.py
────────────
Modular runner for the thesis pipeline. Fixes the "one giant train.py that
re-runs a 1-hour Optuna every time a model at the end crashes" problem.

Each stage:
  • runs independently (pick exactly what you need)
  • SKIPS itself if its result file already exists (unless --force)
  • saves its result to results/stages/<stage>.json so a crash in stage 9
    never costs you stages 1-8 again

USAGE
─────
  # data is cached automatically after first run (force_refresh only if needed)
  python run_stage.py xgboost              # just XGBoost
  python run_stage.py tech_only            # one modality
  python run_stage.py gated                # the key model
  python run_stage.py all                  # everything, but skips done stages
  python run_stage.py all --force          # everything, ignore cache
  python run_stage.py list                 # show stages + which are done

  # run several explicitly:
  python run_stage.py concat cross_attention gated

STAGES
──────
  xgboost, tech_only, fund_only, multimodal, single_encoder,
  concat, gated, regularized, ca_tuned_warmup, ca_reg_warmup, gated_warmup
"""
import os
import sys
import json
import argparse
import traceback

# ── import the existing pipeline functions unchanged ──
from datasets.feature_engineering import prepare_dataset
from train import (
    run_xgboost_pipeline,
    run_modality_ablation,
    run_single_encoder_pipeline,
    run_dual_encoder_pipeline,
    run_gated_cross_attention,
    run_dual_encoder_regularized,
    run_ca_tuned_with_warmup,
    run_ca_regularized_with_warmup,
    run_gated_with_warmup,
)

STAGE_DIR = os.path.join('results', 'stages')
os.makedirs(STAGE_DIR, exist_ok=True)

# stage name -> (callable taking dataset, label)
STAGES = {
    'xgboost':          lambda d: run_xgboost_pipeline(d),
    'tech_only':        lambda d: run_modality_ablation(d, modality='tech_only'),
    'fund_only':        lambda d: run_modality_ablation(d, modality='fund_only'),
    'multimodal':       lambda d: run_modality_ablation(d, modality='multimodal'),
    'single_encoder':   lambda d: run_single_encoder_pipeline(d),
    'concat':           lambda d: run_dual_encoder_pipeline(d, fusion_type='concat'),
    'gated':            lambda d: run_gated_cross_attention(d),
    'regularized':      lambda d: run_dual_encoder_regularized(d, fusion_type='cross_attention'),
    'ca_tuned_warmup':  lambda d: run_ca_tuned_with_warmup(d),
    'ca_reg_warmup':    lambda d: run_ca_regularized_with_warmup(d),
    'gated_warmup':     lambda d: run_gated_with_warmup(d),
}

# canonical order for "all"
ORDER = ['xgboost', 'tech_only', 'fund_only', 'multimodal', 'single_encoder',
         'concat', 'gated', 'regularized', 'ca_tuned_warmup', 'ca_reg_warmup',
         'gated_warmup']


def stage_path(name):
    return os.path.join(STAGE_DIR, f'{name}.json')


def is_done(name):
    return os.path.exists(stage_path(name))


def _extract_metrics(result):
    """Pull test/val metrics out of whatever the pipeline returned."""
    if not isinstance(result, dict):
        return {}
    out = {}
    for k in ('val_metrics', 'test_metrics', 'threshold'):
        if k in result:
            out[k] = result[k]
    # some pipelines return metrics flat
    for k in ('mcc', 'pr_auc', 'roc_auc', 'f1', 'precision', 'recall'):
        if k in result:
            out[k] = result[k]
    return out


def run_one(name, dataset, force=False):
    if name not in STAGES:
        print(f"  ✗ unknown stage: {name}")
        return False
    if is_done(name) and not force:
        print(f"  ⏭  {name}: already done (results/stages/{name}.json) — skipping. "
              f"Use --force to rerun.")
        return True

    print("\n" + "█" * 60)
    print(f"█  STAGE: {name}")
    print("█" * 60)
    try:
        result = STAGES[name](dataset)
        metrics = _extract_metrics(result)
        with open(stage_path(name), 'w') as f:
            json.dump({'stage': name, 'status': 'ok', 'metrics': metrics},
                      f, indent=2, default=str)
        print(f"\n  ✓ {name} done → results/stages/{name}.json")
        return True
    except Exception as e:
        # save the failure so you can see what broke, but DON'T write a
        # success file (so it reruns next time)
        print(f"\n  ✗ {name} FAILED: {e}")
        traceback.print_exc()
        with open(os.path.join(STAGE_DIR, f'{name}.FAILED.txt'), 'w') as f:
            f.write(traceback.format_exc())
        return False


def cmd_list():
    print("\nStages (✓ = done, · = pending):")
    for s in ORDER:
        mark = '✓' if is_done(s) else '·'
        print(f"  {mark} {s}")
    print(f"\nDone files in: {STAGE_DIR}/")


def main():
    ap = argparse.ArgumentParser(description="Modular thesis pipeline runner")
    ap.add_argument('stages', nargs='+',
                    help="stage name(s), or 'all', or 'list'")
    ap.add_argument('--force', action='store_true',
                    help="rerun even if a result file exists")
    ap.add_argument('--refresh-data', action='store_true',
                    help="force_refresh=True (re-fetch from WRDS)")
    args = ap.parse_args()

    if args.stages == ['list']:
        cmd_list()
        return

    # data once, shared across stages
    print("Loading dataset (cache unless --refresh-data)...")
    dataset = prepare_dataset(force_refresh=args.refresh_data)

    targets = ORDER if args.stages == ['all'] else args.stages

    results = {}
    for s in targets:
        results[s] = run_one(s, dataset, force=args.force)

    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    for s in targets:
        status = "✓ ok" if results.get(s) else "✗ failed/skipped"
        print(f"  {s:<18} {status}")
    print(f"\n  Stage results in: {STAGE_DIR}/")
    n_fail = sum(1 for s in targets if not results.get(s))
    if n_fail:
        print(f"  {n_fail} stage(s) failed — rerun just those, others are cached.")
    print("═" * 60)


if __name__ == '__main__':
    main()