#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python run_coverage_age_control.py --self-test-only
python run_coverage_age_control.py --mode smoke --seed 42 --device cuda
for seed in {42..52}; do
  echo "Starting coverage control seed ${seed} at $(date -u +%FT%TZ)"
  python run_coverage_age_control.py --mode train --seed "$seed" --device cuda --allow-candidate-linkage
  echo "Completed seed ${seed} at $(date -u +%FT%TZ)"
done
python evaluate_coverage_age_control.py
