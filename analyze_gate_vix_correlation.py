"""
analyze_gate_vix_correlation.py
──────────────────────────────────
XAI doğrulaması: gated cross-attention'ın kapı değerleri (gate), model
docstring'inde iddia edildiği gibi gerçekten VIX rejimiyle ilişkili mi?

İDDİA (transformer_model.py docstring'i): "Örneğin yüksek VIX rejiminde
fundamental sinyale ağırlık artabilir, sakin dönemde kısılabilir." Bu şu ana
kadar hiç ampirik olarak test edilmedi — sadece teorik bir hipotezdi.

YÖNTEM:
    1. Eğitimli gated_cross_attention modelini test seti üzerinde çalıştır.
    2. Her örnek için CLS-token kapı değerini (fund→tech ve tech→fund
       yönlerinde) çıkar (last_gate_t_full / last_gate_f_full).
    3. Aynı tarihteki VIX_Close ve Is_Yield_Inverted değerleriyle hizala.
    4. Pearson/Spearman korelasyon + yüksek-VIX vs düşük-VIX rejim
       karşılaştırması (medyan split, t-testi).

Sonuç ne çıkarsa çıksın (korelasyon var ya da yok) tez metninde raporlanabilir
— "kapı mekanizması yorumlanabilir bir davranış öğrendi" ya da "kapı
davranışı VIX ile açık bir ilişki göstermiyor, bu da neden performans
kazancı sağlamadığını açıklıyor" şeklinde iki yönlü de savunulabilir.

KULLANIM:
    python analyze_gate_vix_correlation.py \\
        --checkpoint checkpoints/best_dualencoder_gated_cross_attention.pth
"""
import os
import argparse

import numpy as np
import pandas as pd
import torch
from scipy import stats

from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--outdir', type=str, default='results')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    print("[1/3] Dataset + dataloader'lar (cache)...")
    dataset_out = prepare_dataset(force_refresh=False)
    train_loader, val_loader, test_loader, _, (tech_cols, fund_cols) = \
        get_dual_stream_dataloaders(dataset_out, seq_len=20, batch_size=args.batch_size)

    print(f"[2/3] Model yükleniyor (gated_cross_attention): {args.checkpoint}")
    model = DualEncoderTransformer(
        tech_dim=len(tech_cols), fund_dim=len(fund_cols),
        seq_len=20, d_model=64, n_heads=4, n_layers=2, dropout=0.15,
        modality='multimodal', fusion_type='gated_cross_attention',
    )
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print("[3/3] Test setinde kapı değerleri çıkarılıyor...")
    gate_t_vals, gate_f_vals = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_tech = batch['tech_seq'].to(device)
            x_fund = batch['fund_seq'].to(device)
            _ = model(x_tech, x_fund)
            # CLS token pozisyonundaki kapı değeri, d_model boyutu üzerinden
            # ortalanır → örnek başına tek skaler "kapı gücü"
            g_t = model.last_gate_t_full[:, 0, :].mean(dim=-1).cpu().numpy()
            g_f = model.last_gate_f_full[:, 0, :].mean(dim=-1).cpu().numpy()
            gate_t_vals.append(g_t)
            gate_f_vals.append(g_f)

    gate_t_vals = np.concatenate(gate_t_vals)
    gate_f_vals = np.concatenate(gate_f_vals)

    ds = test_loader.dataset
    df = pd.DataFrame({
        'date': ds.dates.values,
        'ticker': ds.tickers,
        'gate_t_fund_to_tech': gate_t_vals,   # tech, fundamental'e ne kadar güveniyor
        'gate_f_tech_to_fund': gate_f_vals,   # fund, technical'e ne kadar güveniyor
    })

    # Aynı tarihteki VIX / yield-inversion değerleri (tüm hisselerde ortak)
    macro_cols = [c for c in ['VIX_Close', 'Is_Yield_Inverted'] if c in dataset_out.columns]
    macro_by_date = dataset_out.groupby(dataset_out.index)[macro_cols].first()
    df = df.merge(macro_by_date, left_on='date', right_index=True, how='left')
    df = df.dropna(subset=macro_cols)

    print("\n" + "=" * 70)
    print("KAPI DEĞERİ — VIX REJİM KORELASYONU")
    print("=" * 70)
    print(f"  {len(df):,} gözlem | {df['date'].nunique():,} benzersiz tarih")

    results_rows = []
    for gate_col, label in [('gate_t_fund_to_tech', 'tech ← fund (tech, fundamental\'e güveni)'),
                            ('gate_f_tech_to_fund', 'fund ← tech (fund, technical\'e güveni)')]:
        print(f"\n  ── {label} ──")
        if 'VIX_Close' in df.columns:
            pear_r, pear_p = stats.pearsonr(df[gate_col], df['VIX_Close'])
            spear_r, spear_p = stats.spearmanr(df[gate_col], df['VIX_Close'])
            print(f"    VIX_Close ile Pearson r={pear_r:+.4f} (p={pear_p:.4g}) | "
                  f"Spearman ρ={spear_r:+.4f} (p={spear_p:.4g})")
            results_rows.append({'gate': gate_col, 'vs': 'VIX_Close', 'method': 'pearson',
                                 'r': pear_r, 'p': pear_p})
            results_rows.append({'gate': gate_col, 'vs': 'VIX_Close', 'method': 'spearman',
                                 'r': spear_r, 'p': spear_p})

            # Medyan-split VIX rejim karşılaştırması
            median_vix = df['VIX_Close'].median()
            high_vix = df[df['VIX_Close'] > median_vix][gate_col]
            low_vix = df[df['VIX_Close'] <= median_vix][gate_col]
            t_stat, t_p = stats.ttest_ind(high_vix, low_vix, equal_var=False)
            print(f"    Yüksek-VIX ort. kapı: {high_vix.mean():.4f} | "
                  f"Düşük-VIX ort. kapı: {low_vix.mean():.4f} | "
                  f"fark={high_vix.mean()-low_vix.mean():+.4f} (Welch t-test p={t_p:.4g})")
            results_rows.append({'gate': gate_col, 'vs': 'VIX_median_split', 'method': 'welch_ttest',
                                 'high_mean': high_vix.mean(), 'low_mean': low_vix.mean(), 'p': t_p})

        if 'Is_Yield_Inverted' in df.columns and df['Is_Yield_Inverted'].nunique() > 1:
            inv = df[df['Is_Yield_Inverted'] == 1][gate_col]
            noninv = df[df['Is_Yield_Inverted'] == 0][gate_col]
            t_stat2, t_p2 = stats.ttest_ind(inv, noninv, equal_var=False)
            print(f"    İnversiyon ort. kapı: {inv.mean():.4f} | "
                  f"Normal ort. kapı: {noninv.mean():.4f} | "
                  f"fark={inv.mean()-noninv.mean():+.4f} (Welch t-test p={t_p2:.4g})")
            results_rows.append({'gate': gate_col, 'vs': 'Yield_Inversion', 'method': 'welch_ttest',
                                 'high_mean': inv.mean(), 'low_mean': noninv.mean(), 'p': t_p2})

    print("\n" + "-" * 70)
    any_significant = any(r.get('p', 1.0) < 0.05 for r in results_rows)
    if any_significant:
        print("  → En az bir ilişki p<0.05 düzeyinde anlamlı: kapı mekanizması VIX/rejim")
        print("    bilgisiyle bir miktar örtüşen bir davranış öğrenmiş — yorumlanabilirlik")
        print("    iddiası ampirik destek buluyor (etki büyüklüğüne dikkat: r küçük olabilir).")
    else:
        print("  → Hiçbir ilişki anlamlı değil: kapı, docstring'de varsayılan VIX-duyarlı")
        print("    davranışı öğrenmemiş görünüyor. Bu da MEVCUT bir bulgu — 'gating')")
        print("    mekanizmasının teorik motivasyonu ampirik olarak doğrulanmadı' diye")
        print("    dürüstçe raporlanabilir.")
    print("=" * 70)

    out_path = os.path.join(args.outdir, 'gate_vix_correlation.csv')
    pd.DataFrame(results_rows).to_csv(out_path, index=False)
    raw_path = os.path.join(args.outdir, 'gate_values_raw.csv')
    df.to_csv(raw_path, index=False)
    print(f"\nSonuçlar kaydedildi → {out_path}")
    print(f"Ham kapı değerleri  → {raw_path}")


if __name__ == '__main__':
    main()