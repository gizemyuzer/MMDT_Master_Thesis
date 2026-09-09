"""
run_xgb_feature_ablation.py
────────────────────────────
XGBoost'u FARKLI ÖZELLİK GRUPLARIYLA koşar — kontrol deneyi.

═══════════════════════════════════════════════════════════════════════
NEDEN BU DENEY ZORUNLU
═══════════════════════════════════════════════════════════════════════
`run_modality_v2.py` sonucu: multi_pure (teknik + 6 firma oranı, MAKROSUZ,
cross-attention) test MCC = 0.1303 ± 0.0017 → mevcut XGBoost'u (0.1100) geçti.

AMA mevcut XGBoost TÜM 51 özelliği alıyor — makro dahil. Yani karşılaştırma
iki şeyi birden değiştiriyor:
    (a) model sınıfı      : ağaç → cross-attention transformer
    (b) özellik kümesi    : makrolu → makrosuz

İki değişkeni birden oynatan bir deney hiçbir şey kanıtlamaz. Jürinin
soracağı ilk soru şudur:

    "Makroyu XGBoost'tan da çıkarsanız o da 0.13'e çıkmaz mıydı?"

Bu script tam olarak o soruyu yanıtlar. Özellik kümesini sabitleyip yalnızca
model sınıfını değiştiren adil bir karşılaştırma sağlar.

OLASI SONUÇLAR:
  XGB(tech+fund) ≈ 0.11  ve  multi_pure = 0.130
      → Füzyon gerçekten kazandı. Aynı özelliklerle transformer daha iyi.
        Tezin ana iddiası kanıtlanmış olur.
  XGB(tech+fund) ≈ 0.13  ve  multi_pure = 0.130
      → Kazanan füzyon değil, MAKRONUN ÇIKARILMASI. Bulgu hâlâ değerli
        ("makro kolonlar her model sınıfında zarar veriyor") ama iddia
        tamamen değişir. Bunu jüriden önce siz bulmalısınız.

═══════════════════════════════════════════════════════════════════════
ADİL TUNING
═══════════════════════════════════════════════════════════════════════
Her özellik kümesi için Optuna araması AYRI çalıştırılır (--trials).
Bunun sebebi: makrolu veri için tune edilmiş parametrelerle makrosuz veriyi
koşmak XGBoost'u haksız yere cezalandırır ve sonucu transformer lehine
saptırır. Kendi bulgunuzu kendiniz sabote etmiş olursunuz.

--trials 0 verilirse cached parametreler kullanılır (hızlı ön okuma; nihai
raporlama için KULLANMAYIN).

KULLANIM:
    python run_xgb_feature_ablation.py                  # 50 trial, tüm gruplar
    python run_xgb_feature_ablation.py --trials 100     # tam arama (yavaş)
    python run_xgb_feature_ablation.py --trials 0       # hızlı ön okuma
    python run_xgb_feature_ablation.py --variants tech_fund all
"""
import os
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    matthews_corrcoef, precision_score, recall_score, f1_score,
)

from datasets.feature_engineering import prepare_dataset, resolve_groups
from models.pytorch_trainer import find_best_threshold_mcc


# ══════════════════════════════════════════════════════════════════
# Özellik kümesi varyantları: ad → (grup listesi, açıklama)
# ══════════════════════════════════════════════════════════════════
VARIANTS = {
    'all': (
        ('tech', 'fund', 'macro', 'interact'),
        'Mevcut XGBoost baseline — 51 özellik (referans: test MCC 0.1100)',
    ),
    'tech_fund': (
        ('tech', 'fund'),
        'multi_pure ile AYNI özellik kümesi → ASIL KONTROL DENEYİ',
    ),
    'tech_fund_xs': (
        ('tech', 'fund', 'fund_xs'),
        'multi_xs ile aynı özellik kümesi',
    ),
    'tech_only': (
        ('tech',),
        'Sadece teknik — fundamental hiç katkı yapıyor mu?',
    ),
    'macro_only': (
        ('macro',),
        'Sadece makro — transformer macro_only ablation\'ının ağaç karşılığı',
    ),
}

TRAIN_END = '2019-12-31'
VAL_START, VAL_END = '2020-01-01', '2021-12-31'
TEST_START, TEST_END = '2022-01-01', '2024-12-31'


def evaluate(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(int)
    y_true = y_true.astype(int)
    out = {}
    try:
        out['roc_auc'] = roc_auc_score(y_true, probs)
    except ValueError:
        out['roc_auc'] = 0.5
    try:
        out['pr_auc'] = average_precision_score(y_true, probs)
    except ValueError:
        out['pr_auc'] = 0.0
    out['accuracy'] = accuracy_score(y_true, y_pred)
    out['mcc'] = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    out['precision'] = precision_score(y_true, y_pred, zero_division=0)
    out['recall'] = recall_score(y_true, y_pred, zero_division=0)
    out['f1'] = f1_score(y_true, y_pred, zero_division=0)
    return out


def run_variant(name, groups, desc, dataset_out, n_trials):
    from models.xgboost_model import train_xgboost

    cols = resolve_groups(dataset_out, groups)
    print("\n" + "▄" * 78)
    print(f"VARYANT: {name}  ({len(cols)} özellik)")
    print(f"  Gruplar : {tuple(groups)}")
    print(f"  {desc}")
    print("▄" * 78)

    tr = dataset_out[dataset_out.index <= TRAIN_END]
    va = dataset_out[(dataset_out.index >= VAL_START) & (dataset_out.index <= VAL_END)]
    te = dataset_out[(dataset_out.index >= TEST_START) & (dataset_out.index <= TEST_END)]
    print(f"  Train: {len(tr):,} | Val: {len(va):,} | Test: {len(te):,}")

    # Median imputation — SADECE train medyanı (val/test'e leak yok)
    med = tr[cols].median()
    Xtr, Xva, Xte = (d[cols].fillna(med).values for d in (tr, va, te))
    ytr, yva, yte = (d['Target'].values for d in (tr, va, te))

    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    # ── Tuning ──
    # n_trials>0 ise bu özellik kümesi için AYRI arama yapılır (adil olan bu).
    # n_trials=0 ise cached parametreler kullanılır (sadece hızlı ön okuma).
    if n_trials > 0:
        import models.xgboost_model as xgbm
        _orig = xgbm._run_optuna_search
        # train_xgboost içinde n_trials=100 sabit; wrapper ile override ediyoruz
        xgbm._run_optuna_search = (
            lambda X, y, dates, spw, n_trials=None, _o=_orig, _n=n_trials:
            _o(X, y, dates, spw, n_trials=_n)
        )
        try:
            model, best_params, (v_roc, v_pr) = train_xgboost(
                Xtr_s, ytr, Xva_s, yva, feature_names=cols,
                train_dates=tr.index, use_cached_params=False,
            )
        finally:
            xgbm._run_optuna_search = _orig
    else:
        print("  [Tuning] --trials 0 → cached parametreler (ÖN OKUMA; "
              "nihai raporlamada kullanma)")
        model, best_params, (v_roc, v_pr) = train_xgboost(
            Xtr_s, ytr, Xva_s, yva, feature_names=cols,
            train_dates=tr.index, use_cached_params=True,
        )

    # ── Eşik SADECE validation'dan seçilir (transformer'larla aynı kriter) ──
    p_va = model.predict_proba(Xva_s)[:, 1]
    p_te = model.predict_proba(Xte_s)[:, 1]
    thr, _ = find_best_threshold_mcc(yva, p_va)

    m_va = evaluate(yva, p_va, thr)
    m_te = evaluate(yte, p_te, thr)

    print(f"\n  VAL  → ROC-AUC {m_va['roc_auc']:.4f} | PR-AUC {m_va['pr_auc']:.4f} "
          f"| MCC {m_va['mcc']:.4f}")
    print(f"  TEST → ROC-AUC {m_te['roc_auc']:.4f} | PR-AUC {m_te['pr_auc']:.4f} "
          f"| MCC {m_te['mcc']:.4f}")
    print(f"  Genelleme farkı (MCC): {m_te['mcc'] - m_va['mcc']:+.4f}")

    row = {'variant': name, 'n_features': len(cols), 'groups': '+'.join(groups),
           'threshold': thr}
    row.update({f'val_{k}': v for k, v in m_va.items()})
    row.update({f'test_{k}': v for k, v in m_te.items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', type=str, nargs='+',
                    default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument('--trials', type=int, default=50,
                    help='Varyant başına Optuna deneme sayısı. 0 = cached params')
    ap.add_argument('--outdir', type=str, default='results')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, 'xgb_feature_ablation.csv')

    print("═" * 78)
    print("XGBOOST ÖZELLİK GRUBU ABLATION — kontrol deneyi")
    print("═" * 78)
    print(f"  Varyantlar : {args.variants}")
    print(f"  Optuna     : varyant başına {args.trials} deneme"
          + ("  ⚠️ CACHED (ön okuma)" if args.trials == 0 else ""))
    print()
    print("  SORU: multi_pure'un (0.1303) üstünlüğü füzyondan mı geliyor,")
    print("        yoksa sadece makronun çıkarılmasından mı?")
    print()

    dataset_out = prepare_dataset(force_refresh=False)

    rows = []
    for name in args.variants:
        groups, desc = VARIANTS[name]
        t0 = time.time()
        try:
            row = run_variant(name, groups, desc, dataset_out, args.trials)
            row['minutes'] = round((time.time() - t0) / 60, 1)
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"  ✓ {name} tamamlandı ({row['minutes']} dk) | kaydedildi")
        except Exception as e:
            print(f"  ✗ {name} BAŞARISIZ: {e}")
            import traceback
            traceback.print_exc()

    if not rows:
        print("\nHiç varyant tamamlanamadı.")
        return

    df = pd.DataFrame(rows)
    print("\n" + "═" * 78)
    print("SONUÇ TABLOSU")
    print("═" * 78)
    show = ['variant', 'n_features', 'val_mcc', 'test_mcc', 'test_pr_auc',
            'test_roc_auc', 'minutes']
    show = [c for c in show if c in df.columns]
    print(df[show].to_string(index=False))

    # ── Karar rehberi ──
    print("\n" + "═" * 78)
    print("YORUM")
    print("═" * 78)
    print("  Transformer referansları (5 seed):")
    print("    multi_pure  test MCC = 0.1303 ± 0.0017   (tech + 6 firma, makrosuz)")
    print("    tech_only   test MCC = 0.1118 ± 0.0145")
    print()

    if 'tech_fund' in df['variant'].values:
        x = float(df.loc[df['variant'] == 'tech_fund', 'test_mcc'].iloc[0])
        print(f"  XGB(tech+fund) test MCC = {x:.4f}")
        print(f"  Fark: multi_pure − XGB(tech+fund) = {0.1303 - x:+.4f}")
        print()
        if x < 0.1303 - 2 * 0.0017:
            print("  ✓ AYNI özellik kümesiyle transformer daha iyi.")
            print("    → Üstünlük FÜZYONDAN geliyor. Tezin ana iddiası ayakta.")
        elif x > 0.1303:
            print("  ⚠️ XGBoost aynı özelliklerle transformer'ı geçiyor.")
            print("    → Kazanan füzyon değil, MAKRONUN ÇIKARILMASI.")
            print("    → İddiayı yeniden çerçevele: 'makro kirliliği her model")
            print("      sınıfında zarar veriyor' — hâlâ değerli ama farklı bir tez.")
        else:
            print("  ~ Fark belirsiz (2 SE bandı içinde).")
            print("    → 'Transformer XGBoost'u geçti' demek yerine 'aynı seviyede,")
            print("      ancak makro çıkarılınca her ikisi de iyileşiyor' demek daha dürüst.")

    if {'all', 'tech_fund'} <= set(df['variant'].values):
        a = float(df.loc[df['variant'] == 'all', 'test_mcc'].iloc[0])
        b = float(df.loc[df['variant'] == 'tech_fund', 'test_mcc'].iloc[0])
        print(f"\n  Makronun XGBoost'a etkisi: {a:.4f} → {b:.4f} ({b - a:+.4f})")
        if b > a:
            print("    → Makro XGBoost'a da zarar veriyor. Bu, 'makro kirliliği'")
            print("      bulgusunun model-sınıfından bağımsız olduğunu gösterir —")
            print("      tezde ayrı bir katkı olarak raporlanabilir.")

    print(f"\nKaydedildi → {out_path}")


if __name__ == '__main__':
    main()