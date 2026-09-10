"""
run_embargo_check.py — sızıntının eşik seçimine etkisini ölçer.

BAĞLAM
    run_purge_audit.py, validation döneminin son 20 işlem gününe ait
    etiketlerin test dönemi fiyatlarıyla belirlendiğini gösterdi (%3.92).
    Eşik (Bölüm 4.12) validation üzerinde seçildiğinden, bu seçim
    kısmen test bilgisi içeriyor olabilir.

    Bu, YENİDEN EĞİTİM GEREKTİRMEDEN ölçülebilir: eşiği bir kez
    embargo'suz, bir kez validation'ın son 20 gününü atarak seçip
    test metriklerini karşılaştırmak yeterlidir.

    Fark ihmal edilebilirse, sızıntının pratik etkisi yoktur ve
    dürüstçe raporlanır. Büyükse, purge'lü yeniden eğitim gerekir.

KULLANIM
    python run_embargo_check.py --pattern "multi_text_seed*.npz"
"""
import os, glob, argparse
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score

VAL_START, VAL_END = '2020-01-01', '2021-12-31'
TEST_START, TEST_END = '2022-01-01', '2024-12-31'


def pick_threshold(y, s, n_grid=300):
    lo, hi = np.percentile(s, [0.5, 99.5])
    best_t, best_m = np.nan, -1.0
    for t in np.linspace(lo, hi, n_grid):
        pred = (s >= t).astype(int)
        if pred.min() == pred.max():
            continue
        m = matthews_corrcoef(y, pred)
        if m > best_m:
            best_m, best_t = m, t
    return best_t, best_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='multi_text_seed*.npz')
    ap.add_argument('--preds-dir', default=os.path.join('results', 'preds_v2'))
    ap.add_argument('--embargo', type=int, default=20,
                    help='validation sonundan atılacak işlem günü sayısı')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.preds_dir, args.pattern)))
    if not files:
        raise SystemExit(f"{args.preds_dir}/{args.pattern} bulunamadı.")

    rows = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if 'split' not in z:
            raise SystemExit(f"{os.path.basename(f)} 'split' içermiyor; "
                             f"export_preds.py --splits test val ile üretin.")
        d = pd.DataFrame({'date': pd.to_datetime(z['dates']),
                          'prob': z['prob'].astype(float),
                          'y': z['target'].astype(int),
                          'split': np.asarray(z['split'])})
        v = d[d.split == 'val']
        t = d[d.split == 'test']
        if v.empty or t.empty:
            print(f"  ⚠️ {os.path.basename(f)}: val veya test boş, atlandı")
            continue

        vdays = np.sort(v['date'].unique())
        cutoff = vdays[-args.embargo] if len(vdays) > args.embargo else vdays[0]
        v_emb = v[v['date'] < cutoff]

        out = {'file': os.path.basename(f).replace('.npz', ''),
               'val_n': len(v), 'val_n_embargo': len(v_emb),
               'atilan_%': 100 * (1 - len(v_emb) / len(v))}

        for lab, vv in (('full', v), ('embargo', v_emb)):
            thr, vmcc = pick_threshold(vv['y'].values, vv['prob'].values)
            pred = (t['prob'].values >= thr).astype(int)
            out[f'{lab}_thr'] = thr
            out[f'{lab}_val_mcc'] = vmcc
            out[f'{lab}_test_mcc'] = (matthews_corrcoef(t['y'].values, pred)
                                      if pred.min() != pred.max() else 0.0)
            out[f'{lab}_test_prec'] = precision_score(t['y'].values, pred,
                                                      zero_division=0)
            out[f'{lab}_test_rec'] = recall_score(t['y'].values, pred,
                                                  zero_division=0)
        rows.append(out)

    R = pd.DataFrame(rows)
    R['d_test_mcc'] = R['embargo_test_mcc'] - R['full_test_mcc']
    R['d_thr'] = R['embargo_thr'] - R['full_thr']
    R.to_csv('results/embargo_check.csv', index=False)

    print("═" * 74)
    print(f"EMBARGO KONTROLÜ — validation sonundan {args.embargo} gün atıldı")
    print("═" * 74)
    print(f"  atılan validation gözlemi: %{R['atilan_%'].mean():.2f}")
    print()
    print(f"{'':<10}{'eşik':>10}{'val MCC':>10}{'test MCC':>10}"
          f"{'prec':>9}{'rec':>9}")
    print("-" * 60)
    for lab in ('full', 'embargo'):
        print(f"{lab:<10}{R[f'{lab}_thr'].mean():>10.4f}"
              f"{R[f'{lab}_val_mcc'].mean():>10.4f}"
              f"{R[f'{lab}_test_mcc'].mean():>10.4f}"
              f"{R[f'{lab}_test_prec'].mean():>9.4f}"
              f"{R[f'{lab}_test_rec'].mean():>9.4f}")
    print("-" * 60)
    dm = R['d_test_mcc']
    print(f"\nΔ test MCC (embargo − full): {dm.mean():+.4f} ± {dm.std(ddof=1):.4f}  "
          f"({int((dm > 0).sum())}/{len(dm)} pozitif)")
    print(f"Δ eşik: {R['d_thr'].mean():+.4f}")

    print()
    print("YORUM")
    if abs(dm.mean()) < 0.002:
        print("  Fark ihmal edilebilir (<0.002 MCC). Validation sızıntısının")
        print("  eşik seçimi üzerinden pratik bir etkisi yok. Sızıntı")
        print("  büyüklüğü raporlanır, sonuçlar geçerli kalır.")
    elif abs(dm.mean()) < 0.01:
        print("  Fark küçük ama sıfır değil. Embargo'lu eşik ana sonuç")
        print("  olarak raporlanmalı, embargo'suz olan duyarlılık kontrolü.")
    else:
        print("  Fark maddi. Embargo'lu eşik ZORUNLU; ayrıca purge'lü")
        print("  yeniden eğitim gerekebilir (erken durdurma da etkileniyor).")
    print("\n→ results/embargo_check.csv")


if __name__ == '__main__':
    main()