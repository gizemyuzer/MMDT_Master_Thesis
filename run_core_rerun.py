"""
run_core_rerun.py
────────────────────
Delisting-return + Altman Z-score zenginleştirmesinden sonra, SADECE bilgi
değeri olan 4 pipeline'ı yeniden çalıştırır — train.py'deki 12 aşamanın
TAMAMINI değil.

NEDEN SADECE BU 4'Ü:
  - XGBoost        : ana baseline, feature seti değiştiği için güncellenmeli
  - tech_only      : referans (fundamental'siz), değişmemesi beklenir
  - fund_only      : EN KRİTİK — Altman Z gerçekten sinyal katıyorsa, bu
                      pipeline'ın MCC'si burada görünür (eskiden ~-0.006,
                      test'te çöküyordu). "Fundamental modalite ince"
                      eleştirisine en direkt ampirik cevap.
  - multimodal     : cross-attention füzyonlu referans satır

ATLANANLAR (bilinçli): regularized, ca_tuned_warmup, ca_reg_warmup,
gated_warmup, single_encoder — bunlar daha önce ESKİ veriyle denenip
başarısız olmuştu, sebebi mimari/hyperparameter kaynaklıydı (overfitting,
eski Optuna params), fundamental veri zenginliğiyle ilgisizdi. Yeniden
koşturmak zaman kaybı. concat ve gated_cross_attention multi-seed
karşılaştırması için run_multiseed.py kullanılmalı (bu script'te değil).

ÖNKOŞUL: Bu script'i çalıştırmadan ÖNCE veriyi bir kez yenile:
    python -c "from datasets.feature_engineering import prepare_dataset; prepare_dataset(force_refresh=True)"
Aksi halde hâlâ eski cache'i okur, hiçbir şey değişmemiş görünür.

KULLANIM:
    python run_core_rerun.py
"""
import os
import time
import argparse

import pandas as pd

from datasets.feature_engineering import prepare_dataset
from train import run_xgboost_pipeline, run_modality_ablation

ALL_PIPELINES = ['XGBoost', 'tech_only', 'fund_only', 'multimodal_cross_attn']


def _flatten(name, result):
    """XGBoost ve modality-ablation sonuçlarını tek satırlık dict'e indirger."""
    row = {'pipeline': name}
    for split in ('val', 'test'):
        m = result.get(f'{split}_metrics')
        if not m:
            continue
        for k, v in m.items():
            row[f'{split}_{k}'] = v
    if result.get('threshold') is not None:
        row['threshold'] = result['threshold']
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipelines', type=str, nargs='+', default=ALL_PIPELINES,
                    choices=ALL_PIPELINES,
                    help="Sadece belirtilenleri çalıştır, ör: --pipelines fund_only multimodal_cross_attn")
    ap.add_argument('--yes', action='store_true',
                    help="Sağlık kontrolü uyarısında soru sorma, otomatik devam et "
                         "(gözetimsiz/gece koşuları için — bkz. run_all.py)")
    args = ap.parse_args()

    outdir = 'results'
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, 'core_rerun_summary.csv')

    print("═" * 78)
    print("CORE RERUN — delisting-return + Altman Z sonrası temel karşılaştırma")
    print("═" * 78)
    print(f"  Bu koşuda çalıştırılacaklar: {args.pipelines}")
    print("  ÖNKOŞUL: prepare_dataset(force_refresh=True) daha önce çalıştırılmış olmalı")
    print()

    # ── Önceki sonuçları koru — sadece bu koşuda çalıştırılan pipeline'ların
    # satırları güncellenir, geri kalanı (ör. XGBoost, tech_only) dokunulmadan kalır ──
    existing_rows = {}
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        existing_rows = {r['pipeline']: r for r in prev.to_dict('records')}
        kept = [p for p in existing_rows if p not in args.pipelines]
        if kept:
            print(f"  [KORUNAN] Önceki sonuçlar dokunulmadan tutulacak: {kept}")

    print("[1/2] Dataset yükleniyor (cache — yenilenmiş olmalı)...")
    dataset_out = prepare_dataset(force_refresh=False)

    # Sağlık kontrolü: yeni fundamental kolonlar gerçekten cache'te mi?
    new_cols = ['Altman_Z', 'Retained_Earnings_TA', 'Market_Value_to_Liab']
    missing = [c for c in new_cols if c not in dataset_out.columns]
    if missing:
        print(f"\n  ⚠️  UYARI: {missing} cache'te YOK. Muhtemelen force_refresh=True henüz "
              f"çalıştırılmadı — bu koşu ESKİ veriyle yapılacak, karşılaştırma anlamsız olur.")
        if args.yes:
            print("  --yes verildi → soru sorulmadan devam ediliyor.")
        else:
            resp = input("  Yine de devam et? (e/h): ").strip().lower()
            if resp != 'e':
                print("  İptal edildi. Önce veriyi yenile.")
                return
    else:
        print(f"  ✓ Yeni fundamental kolonlar cache'te mevcut: {new_cols}")

    print("\n[2/2] Pipeline'lar koşuluyor...\n")

    all_pipeline_fns = {
        'XGBoost': lambda: run_xgboost_pipeline(dataset_out),
        'tech_only': lambda: run_modality_ablation(dataset_out, modality='tech_only'),
        'fund_only': lambda: run_modality_ablation(dataset_out, modality='fund_only'),
        'multimodal_cross_attn': lambda: run_modality_ablation(dataset_out, modality='multimodal'),
    }

    # Sonuç sözlüğü: önce korunan (rerun edilmeyen) satırlarla başla
    results_by_name = {p: r for p, r in existing_rows.items() if p not in args.pipelines}

    for name in args.pipelines:
        fn = all_pipeline_fns[name]
        print("\n" + "▄" * 78)
        print(f"PIPELINE: {name}")
        print("▄" * 78)
        t0 = time.time()
        try:
            result = fn()
            elapsed = round((time.time() - t0) / 60, 1)
            row = _flatten(name, result)
            row['minutes'] = elapsed
            results_by_name[name] = row
            pd.DataFrame(list(results_by_name.values())).to_csv(out_path, index=False)
            print(f"\n  ✓ {name} tamamlandı ({elapsed} dk) | "
                  f"test MCC={row.get('test_mcc', float('nan')):.4f} | kaydedildi")
        except Exception as e:
            print(f"\n  ✗ {name} BAŞARISIZ: {e}")
            import traceback
            traceback.print_exc()

    if not results_by_name:
        print("\nHiçbir pipeline tamamlanamadı.")
        return

    df = pd.DataFrame(list(results_by_name.values()))
    df.to_csv(out_path, index=False)

    print("\n" + "═" * 78)
    print("SONUÇ TABLOSU (test seti)")
    print("═" * 78)
    cols_to_show = ['pipeline', 'test_mcc', 'test_pr_auc', 'test_roc_auc', 'test_f1', 'minutes']
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    print(df[cols_to_show].to_string(index=False))

    if 'fund_only' in df['pipeline'].values and 'test_mcc' in df.columns:
        fo_mcc = df.loc[df['pipeline'] == 'fund_only', 'test_mcc'].iloc[0]
        print(f"\n  fund_only test MCC: {fo_mcc:.4f} "
              f"(eski veride ~-0.006 idi — pozitifse Altman Z gerçek sinyal katmış demektir)")

    print(f"\nÖzet kaydedildi → {out_path}")
    print(f"\nSonraki adım: concat/gated_cross_attention/static_context/film/late_fusion "
          f"için run_multiseed.py ve run_late_fusion.py kullan.")


if __name__ == '__main__':
    main()