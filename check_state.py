"""Hangi degisikliklerin uygulandigini kontrol eder. python check_state.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

CONTENT = [
    ('datasets/feature_engineering.py', 'attach_text_features', 'metin modalitesi (attach)'),
    ('datasets/feature_engineering.py', 'TEXT_COLS',            'metin kolon grubu'),
    ('run_modality_v2.py',              'ABLATION_FUSION',      'fuzyon override'),
    ('run_modality_v2.py',              'multi_text',           'metin hucreleri'),
    ('run_modality_v2.py',              '_backup.csv',          '--force veri kaybi duzeltmesi'),
    ('run_xgb_factorial.py',            'I+II+IV',              'XGB metin hucreleri'),
    ('models/transformer_model.py',     'capture_attribution',  'attribution yakalama'),
    ('run_portfolio_simulation.py',     'rebalance_days',       'exclude_pct kaydi'),
]

FILES = [
    'build_text_features.py', 'fetch_edgar_filings.py', 'run_bear_market.py',
    'run_attribution.py', 'run_skill_curve.py', 'diagnose_gpu.py',
]

print(f"Klasor: {HERE}\n")
print("--- Dosya icindeki degisiklikler ---")
for rel, pat, desc in CONTENT:
    p = os.path.join(HERE, rel)
    if not os.path.exists(p):
        print(f"  DOSYA YOK  {desc}  ({rel})")
        continue
    with open(p, encoding='utf-8', errors='ignore') as f:
        ok = pat in f.read()
    print(f"  {'VAR  ' if ok else 'EKSIK'}      {desc}")

print("\n--- Dosya var mi ---")
for rel in FILES:
    exists = os.path.exists(os.path.join(HERE, rel))
    print(f"  {'VAR  ' if exists else 'EKSIK'}      {rel}")