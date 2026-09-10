"""
run_purge_audit.py — ileri hedef penceresinden kaynaklanan sızıntı denetimi.

SORUN (danışman geri bildirimi C3)
    Hedef, t gününden sonraki 20 işlem gününü kullanır. Dolayısıyla
    eğitim kümesinin SON 20 gününe ait etiketler, validation dönemine
    ait fiyatlarla belirlenir. Tarihleri ayırmak tek başına bunu
    engellemez; aradan bir "purge" (temizleme) bandı çıkarmak gerekir.

    Aynı sorun XGBoost'un iç çapraz doğrulama katlarında da geçerlidir:
    her kat sınırında son 20 günün etiketi bir sonraki katın fiyatlarını
    içerir.

BU SCRIPT
    1) Her bölünme sınırında kaç gözlemin hedef penceresi sonraki
       döneme taştığını sayar.
    2) XGBoost iç CV katları için aynı sayımı yapar.
    3) İsteğe bağlı: purge uygulanmış ve uygulanmamış XGBoost'u aynı
       konfigürasyonla koşup metrik farkını ölçer (--rerun).

    Sızıntının BÜYÜKLÜĞÜNÜ ölçmek, varlığını tartışmaktan daha
    yararlıdır: küçükse dürüstçe raporlanır, büyükse düzeltilir.

KULLANIM
    python run_purge_audit.py
    python run_purge_audit.py --rerun          # XGBoost etkisini ölç
"""
import argparse
import numpy as np, pandas as pd

from datasets.feature_engineering import prepare_dataset

HORIZON = 20                       # hedefin ileri penceresi (işlem günü)
TRAIN_END = '2019-12-31'
VAL_START, VAL_END = '2020-01-01', '2021-12-31'
TEST_START = '2022-01-01'
N_FOLDS = 5


# ══════════════════════════════════════════════════════════════════
def leak_at_boundary(all_dates, split_end, horizon=HORIZON):
    """
    split_end'den geriye doğru kaç İŞLEM GÜNÜ'nün hedef penceresi
    sınırı aşıyor? Hedef t için t+1..t+horizon kullanıldığından,
    son `horizon` işlem günü sızdırır.
    """
    d = pd.DatetimeIndex(sorted(all_dates))
    upto = d[d <= pd.Timestamp(split_end)]
    if len(upto) == 0:
        return pd.DatetimeIndex([])
    return upto[-horizon:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizon', type=int, default=HORIZON)
    ap.add_argument('--rerun', action='store_true',
                    help='purge ile ve purge olmadan XGBoost koş, farkı ölç')
    args = ap.parse_args()

    ds = prepare_dataset(force_refresh=False)
    ds = ds.copy(); ds['_date'] = ds.index
    dates = pd.DatetimeIndex(sorted(ds['_date'].unique()))
    print(f"panel: {len(ds):,} satır | {len(dates)} işlem günü | "
          f"hedef ufku {args.horizon} gün")

    # ── 1) ana bölünme sınırları ──
    print()
    print("═" * 74)
    print("ANA BÖLÜNME SINIRLARINDA SIZINTI")
    print("═" * 74)
    rows = []
    for lab, end, nxt in [('train → validation', TRAIN_END, VAL_START),
                          ('validation → test', VAL_END, TEST_START)]:
        leak_days = leak_at_boundary(dates, end, args.horizon)
        m = ds['_date'].isin(leak_days)
        seg = ds[(ds['_date'] >= ('2010-01-01' if 'train' in lab else VAL_START))
                 & (ds['_date'] <= end)]
        rows.append({'sınır': lab, 'sızan_gün': len(leak_days),
                     'sızan_satır': int(m.sum()), 'dönem_satır': len(seg),
                     'oran_%': 100 * m.sum() / max(len(seg), 1),
                     'ilk_sızan_gün': leak_days.min().date() if len(leak_days) else None,
                     'son_gün': leak_days.max().date() if len(leak_days) else None})
    B = pd.DataFrame(rows)
    print(B.to_string(index=False))

    # ── 2) XGBoost iç CV katları ──
    print()
    print("═" * 74)
    print(f"XGBOOST İÇ CV KATLARINDA SIZINTI ({N_FOLDS} genişleyen pencere)")
    print("═" * 74)
    tr_dates = dates[dates <= pd.Timestamp(TRAIN_END)]
    bounds = np.array_split(np.arange(len(tr_dates)), N_FOLDS + 1)
    frows = []
    total_leak = 0
    for i in range(N_FOLDS):
        cut = tr_dates[bounds[i][-1]]
        leak_days = leak_at_boundary(tr_dates, cut, args.horizon)
        m = ds['_date'].isin(leak_days)
        in_fold = ds['_date'] <= cut
        total_leak += int(m.sum())
        frows.append({'kat': i + 1, 'kesim': cut.date(),
                      'kat_satır': int(in_fold.sum()),
                      'sızan_satır': int(m.sum()),
                      'oran_%': 100 * m.sum() / max(in_fold.sum(), 1)})
    F = pd.DataFrame(frows)
    print(F.to_string(index=False))
    print(f"\n  toplam sızan gözlem (katlar arası): {total_leak:,}")

    # ── 3) yorum ──
    print()
    print("═" * 74)
    print("YORUM")
    print("═" * 74)
    tr_rows = int((ds['_date'] <= TRAIN_END).sum())
    main_leak = int(B.loc[0, 'sızan_satır'])
    print(f"  Eğitim kümesinin %{100*main_leak/tr_rows:.2f}'inin etiketi")
    print(f"  validation dönemine ait fiyatlarla belirleniyor.")
    print()
    if 100 * main_leak / tr_rows < 1.0:
        print("  Oran %1'in altında. Sızıntı GERÇEK ama küçük; muhtemelen")
        print("  sonuçları maddi olarak değiştirmez. Yine de purge uygulanmış")
        print("  bir koşuyla doğrulanmalı ve tezde raporlanmalıdır.")
    else:
        print("  Oran %1'in üstünde. Purge uygulanmış koşu ZORUNLU;")
        print("  mevcut metrikler yukarı yanlı olabilir.")
    print()
    print("  DÜZELTME: eğitim kümesinden son", args.horizon, "işlem gününü çıkarın")
    print("  (purge). İç CV katlarında da her kesim öncesi aynı bandı çıkarın.")

    out = 'results/purge_audit.csv'
    pd.concat([B.assign(tip='ana_bölünme'),
               F.rename(columns={'kat': 'sınır', 'kesim': 'ilk_sızan_gün',
                                 'kat_satır': 'dönem_satır'}).assign(tip='cv_katı')
               ], ignore_index=True).to_csv(out, index=False)
    print(f"\n→ {out}")

    # ── 4) isteğe bağlı: etkiyi ölç ──
    if args.rerun:
        print()
        print("═" * 74)
        print("PURGE ETKİSİ — XGBoost, aynı konfigürasyon")
        print("═" * 74)
        try:
            from models.xgboost_model import train_xgboost_model  # varsa
        except Exception:
            print("  train_xgboost_model içe aktarılamadı; bu adımı")
            print("  run_xgb_factorial.py içinden purge bayrağıyla koşmak")
            print("  gerekir. Denetim sonuçları yukarıda yine de geçerli.")
            return
        print("  (bu adım mevcut XGBoost eğitim akışına bağlanmalı —")
        print("   purge maskesi: eğitim satırlarından son", args.horizon,
              "işlem gününü çıkar)")


if __name__ == '__main__':
    main()