"""
run_attribution.py
───────────────────
TRANSFORMER YORUMLANABİLİRLİĞİ — dört bağımsız attribution kanalı.

═══════════════════════════════════════════════════════════════════════
NEDEN BU SCRIPT
═══════════════════════════════════════════════════════════════════════
Proje tanımı "feature-attribution analyses (e.g., SHAP)" istiyor. Şu ana kadar
SHAP yalnızca XGBoost için üretildi; derin füzyon modelleri için hiçbir
attribution yok. Bu, tezin en gözle görülür eksiği: mimarinin ne öğrendiğine
dair iddialar var ama ölçüm yok.

SHAP'i transformer'a doğrudan uygulamak iki nedenle tercih edilmedi:
  (1) KernelSHAP dizisel girdide (20 gün × 38 özellik = 760 boyut) pratik
      değil; DeepSHAP ise MultiheadAttention için güvenilir kural üretmiyor.
  (2) Daha önemlisi: XGBoost SHAP'i ile karşılaştırılabilir olması gerekiyor.
      Permütasyon önemi HER İKİ model ailesine de aynı biçimde uygulanabilir,
      yani "hangi modalite hangi modelde önemli" sorusu adil sorulabilir.

Bu yüzden dört kanal:

  [1] PERMÜTASYON ÖNEMİ — model-agnostik, XGBoost ile birebir karşılaştırılabilir.
      Bir özelliğin değerleri örnekler arası karıştırılır; performans düşüşü
      o özelliğin katkısıdır. Hem tek tek özellik hem de MODALİTE GRUBU
      düzeyinde hesaplanır. Grup düzeyi, ablation sonuçlarının bağımsız bir
      doğrulaması olur: eğer I+II modelinde fund grubunun permütasyon önemi
      ≈ 0 çıkarsa, Δ(II|I) ≈ 0 bulgusu ikinci bir yöntemle teyit edilmiş olur.

  [2] KAPI (GATE) ANALİZİ — gated_cross_attention'ın öğrenilmiş kapı değerleri.
      transformer_model.py docstring'i "yüksek VIX rejiminde fundamental
      sinyale ağırlık artabilir" diyor. Bu iddia ampirik olarak test edilir.
      KRİTİK: bu analiz MAKROSUZ (I+II) özellik kümesiyle koşulan checkpoint'ler
      üzerinde yapılır. Daha önce raporlanan kapı=0.4892 değeri makro içeren
      eski kurulumdan geliyordu ve tezin geri kalanıyla tutarsızdı.

  [3] CROSS-ATTENTION AĞIRLIK PROFİLİ — tech akışının CLS token'ı, fund
      akışının hangi zaman adımlarına bakıyor? Düzgün bir dikkat profili
      (ör. son günlere yoğunlaşma) mimarinin anlamlı bir şey öğrendiğini,
      düz/uniform bir profil ise cross-attention'ın fiilen devre dışı
      kaldığını gösterir.

  [4] FiLM GAMMA — fundamental akışın technical temsili ölçekleme katsayısı.
      gamma < 1: fundamental bağlam technical sinyali BASTIRIYOR.

Her kanal seed'ler arası ortalama ± std olarak raporlanır; tek seed'e
güvenilmez.

═══════════════════════════════════════════════════════════════════════
METODOLOJİ NOTLARI
═══════════════════════════════════════════════════════════════════════
EŞİK: her seed için yalnızca validation'dan MCC-optimal eşik seçilir, test'e
bakılarak hiçbir karar verilmez. Permütasyon önemi hem eşiğe bağlı (MCC) hem
eşikten bağımsız (PR-AUC) metrikle raporlanır; PR-AUC birincil kabul edilir
çünkü eşik seçimi kaynaklı gürültüden etkilenmez.

PERMÜTASYON ŞEKLİ: bir özelliğin (N, T) bloğunun tamamı birlikte karıştırılır,
zaman adımları ayrı ayrı değil. Böylece özelliğin kendi içindeki zamansal
otokorelasyonu korunur, sadece hangi örneğe ait olduğu bozulur. Zaman
adımlarını da karıştırmak, ölçülen düşüşü "özellik önemi" değil "zamansal
yapı önemi" ile karıştırırdı.

GRUP PERMÜTASYONU: bir gruptaki tüm kolonlar AYNI permütasyon indeksiyle
karıştırılır. Farklı indeks kullanmak grup içi korelasyonu da yok ederdi ve
önemi yapay olarak şişirirdi.

KULLANIM:
    python run_attribution.py                          # uygulanabilir tüm kanallar
    python run_attribution.py --channels permutation   # sadece permütasyon
    python run_attribution.py --repeats 5              # permütasyon tekrarı
    python run_attribution.py --groups-only            # hızlı: sadece grup düzeyi
    python run_attribution.py --self-test              # sentetik bilinen-cevap testi

Çıktılar: visualization/attribution/*.png  +  results/attr_*.csv
"""
import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, matthews_corrcoef
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer
from models.pytorch_trainer import find_best_threshold_mcc
from run_modality_v2 import ABLATIONS, BASE_CONFIG, fusion_for

OUT_FIG = os.path.join('visualization', 'attribution')
OUT_CSV = 'results'
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_CSV, exist_ok=True)

C = {'tech': '#2E86AB', 'fund': '#A23B72', 'gray': '#666666', 'accent': '#F18F01'}

# Hangi checkpoint hangi kanalı besler
CHANNEL_ABLATION = {
    'permutation': 'multi_pure',        # cross_attention, I+II
    'attention': 'multi_pure',          # aynı checkpoint
    'gate': 'multi_pure_gated',         # gated_cross_attention, I+II
    'film': 'multi_pure_film',          # FiLM, I+II
}


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def save_fig(fig, name):
    p = os.path.join(OUT_FIG, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    → {p}")


def ckpt_path(ablation, seed):
    return os.path.join('checkpoints', f'best_v2_{ablation}_seed{seed}.pth'.lower())


def build_model(ablation, tech_dim, fund_dim):
    _, _, modality, _ = ABLATIONS[ablation]
    return DualEncoderTransformer(
        tech_dim=tech_dim, fund_dim=fund_dim,
        seq_len=BASE_CONFIG['seq_len'], d_model=BASE_CONFIG['d_model'],
        n_heads=BASE_CONFIG['n_heads'], n_layers=BASE_CONFIG['n_layers'],
        dropout=BASE_CONFIG['dropout'],
        modality=modality, fusion_type=fusion_for(ablation),
    )


@torch.no_grad()
def predict(model, dataset, device, batch_size=512):
    """Bir dataset üzerinde olasılık tahminleri — sırayı korur."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    out = []
    for batch in loader:
        logits = model(batch['tech_seq'].to(device), batch['fund_seq'].to(device))
        out.append(torch.sigmoid(logits).float().cpu().numpy().ravel())
    return np.concatenate(out)


def score(y_true, prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    return {
        'pr_auc': float(average_precision_score(y_true, prob)),
        'mcc': float(matthews_corrcoef(y_true, (prob >= threshold).astype(int))),
    }


# ══════════════════════════════════════════════════════════════════
# [1] PERMÜTASYON ÖNEMİ
# ══════════════════════════════════════════════════════════════════
def permutation_importance(model, dataset, device, threshold, tech_cols, fund_cols,
                           repeats=3, groups_only=False, rng_seed=0, batch_size=512):
    """
    Özellik ve grup düzeyinde permütasyon önemi.

    Dönen değer: DataFrame(target, kind, stream, drop_pr_auc, drop_mcc, ...)
    'drop' = taban performans − permütasyon sonrası performans.
    POZİTİF drop = özellik faydalı. NEGATİF drop = özellik zararlı (gürültü);
    bu tezde beklenen bir sonuç, çünkü makro değişkenlerin zarar verdiği
    ablation'da zaten gösterildi.
    """
    y = dataset.labels
    base_prob = predict(model, dataset, device, batch_size)
    base = score(y, base_prob, threshold)
    print(f"    Taban: PR-AUC={base['pr_auc']:.4f} | MCC={base['mcc']:.4f}")

    # (dizi_adı, kolon_indeksleri, etiket, akış)
    targets = [
        ('tech_sequences', list(range(len(tech_cols))), 'GRUP: teknik (I)', 'tech'),
        ('fund_sequences', list(range(len(fund_cols))), 'GRUP: fundamental (II)', 'fund'),
    ]
    if not groups_only:
        targets += [('tech_sequences', [j], name, 'tech')
                    for j, name in enumerate(tech_cols)]
        targets += [('fund_sequences', [j], name, 'fund')
                    for j, name in enumerate(fund_cols)]

    rows = []
    n = len(y)
    for k, (arr_name, idxs, label, stream) in enumerate(targets, 1):
        arr = getattr(dataset, arr_name)
        original = arr[:, :, idxs].copy()          # (N, T, len(idxs))
        pr_drops, mcc_drops = [], []
        for r in range(repeats):
            rng = np.random.default_rng(rng_seed + 1000 * r)
            perm = rng.permutation(n)
            # Grup içi korelasyonu korumak için TEK permütasyon indeksi
            arr[:, :, idxs] = original[perm]
            prob = predict(model, dataset, device, batch_size)
            s = score(y, prob, threshold)
            pr_drops.append(base['pr_auc'] - s['pr_auc'])
            mcc_drops.append(base['mcc'] - s['mcc'])
        arr[:, :, idxs] = original                 # geri yükle — ZORUNLU
        rows.append({
            'target': label, 'stream': stream,
            'kind': 'group' if len(idxs) > 1 else 'feature',
            'drop_pr_auc': float(np.mean(pr_drops)),
            'drop_pr_auc_std': float(np.std(pr_drops, ddof=1)) if repeats > 1 else 0.0,
            'drop_mcc': float(np.mean(mcc_drops)),
            'drop_mcc_std': float(np.std(mcc_drops, ddof=1)) if repeats > 1 else 0.0,
            'base_pr_auc': base['pr_auc'], 'base_mcc': base['mcc'],
        })
        print(f"    [{k}/{len(targets)}] {label:<28} ΔPR-AUC={rows[-1]['drop_pr_auc']:+.5f}")

    # Geri yükleme doğrulaması — sessiz veri bozulmasına karşı
    check = predict(model, dataset, device, batch_size)
    if not np.allclose(check, base_prob, atol=1e-5):
        raise RuntimeError(
            "Permütasyon sonrası dataset geri yüklenemedi — sonuçlar geçersiz. "
            "Bu bir kod hatasıdır, çıktıyı KULLANMAYIN."
        )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# [2] KAPI + [3] ATTENTION + [4] FiLM — forward sırasında yakala
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def capture_internals(model, dataset, device, batch_size=512):
    """
    Kapı, attention ağırlıkları ve FiLM katsayılarını örnek bazında toplar.
    Sadece CLS pozisyonu saklanır — (B, T+1, D) tensörünün tamamı gereksiz
    büyük ve yorumu belirsizdir; karar CLS üzerinden veriliyor.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    model.capture_attribution = True

    gate_t, gate_f, gamma, attn_prof = [], [], [], []
    for batch in loader:
        model(batch['tech_seq'].to(device), batch['fund_seq'].to(device))
        if model.last_gate_t_full is not None:
            gate_t.append(model.last_gate_t_full[:, 0, :].mean(-1).float().cpu().numpy())
            gate_f.append(model.last_gate_f_full[:, 0, :].mean(-1).float().cpu().numpy())
        if model.last_film_gamma_full is not None:
            gamma.append(model.last_film_gamma_full.mean(-1).float().cpu().numpy())
        if model.last_attn_t2f is not None:
            # CLS satırı: tech CLS'in fund token'larına dağıttığı dikkat
            attn_prof.append(model.last_attn_t2f[:, 0, :].float().cpu().numpy())

    model.capture_attribution = False
    out = {}
    if gate_t:
        out['gate_t'] = np.concatenate(gate_t)
        out['gate_f'] = np.concatenate(gate_f)
    if gamma:
        out['film_gamma'] = np.concatenate(gamma)
    if attn_prof:
        out['attn_t2f'] = np.concatenate(attn_prof)   # (N, T+1)
    return out


def gate_regime_analysis(df, gate_cols):
    """Kapı değerleri ile VIX / yıl rejimi ilişkisi."""
    rows = []
    for col in gate_cols:
        if col not in df.columns:
            continue
        if 'VIX_Close' in df.columns:
            r, p = stats.spearmanr(df[col], df['VIX_Close'])
            rows.append({'gate': col, 'vs': 'VIX_Close', 'stat': 'spearman_rho',
                         'value': r, 'p': p})
            med = df['VIX_Close'].median()
            hi = df.loc[df['VIX_Close'] > med, col]
            lo = df.loc[df['VIX_Close'] <= med, col]
            t, p2 = stats.ttest_ind(hi, lo, equal_var=False)
            rows.append({'gate': col, 'vs': 'VIX_median_split', 'stat': 'mean_diff',
                         'value': hi.mean() - lo.mean(), 'p': p2,
                         'high_mean': hi.mean(), 'low_mean': lo.mean()})
        for yr in sorted(df['year'].unique()):
            sub = df.loc[df['year'] == yr, col]
            rows.append({'gate': col, 'vs': f'year_{yr}', 'stat': 'mean',
                         'value': sub.mean(), 'p': np.nan})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# SENTETİK BİLİNEN-CEVAP TESTİ
# ══════════════════════════════════════════════════════════════════
def self_test():
    """
    Permütasyon öneminin GERÇEKTEN önemli özelliği bulduğunu doğrular.

    Kurgu: rastgele bir dataset üretilir, hedef TEK bir teknik kolondan
    (indeks 3) deterministik olarak türetilir. Eğitilmemiş ama elle kurulmuş
    bir model o kolonu okur. Beklenen: kolon 3'ün önemi yüksek, diğerlerinin
    ≈ 0. Bu tutmazsa permütasyon kodunda hata var demektir.
    """
    print("\n" + "═" * 78)
    print("SENTETİK BİLİNEN-CEVAP TESTİ")
    print("═" * 78)

    class Toy:
        def __init__(self, n=2000, T=20, dt=6, df_=4):
            rng = np.random.default_rng(0)
            self.tech_sequences = rng.normal(size=(n, T, dt)).astype(np.float32)
            self.fund_sequences = rng.normal(size=(n, T, df_)).astype(np.float32)
            signal = self.tech_sequences[:, -1, 3]
            self.labels = (signal > 0).astype(np.float32)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {'tech_seq': torch.tensor(self.tech_sequences[i]),
                    'fund_seq': torch.tensor(self.fund_sequences[i]),
                    'label': torch.tensor(self.labels[i])}

    class Oracle(torch.nn.Module):
        """Kolon 3'ün son gününü okuyup logit üreten sahte model."""
        capture_attribution = False
        last_gate_t_full = last_gate_f_full = last_film_gamma_full = last_attn_t2f = None

        def forward(self, x_tech, x_fund):
            return (x_tech[:, -1, 3] * 8.0).unsqueeze(-1)

    ds, model, dev = Toy(), Oracle(), torch.device('cpu')
    imp = permutation_importance(
        model, ds, dev, threshold=0.5,
        tech_cols=[f'tech_{i}' for i in range(6)],
        fund_cols=[f'fund_{i}' for i in range(4)],
        repeats=2, groups_only=False, batch_size=512,
    )
    feats = imp[imp['kind'] == 'feature'].set_index('target')['drop_pr_auc']
    signal_drop = feats['tech_3']
    others = feats.drop('tech_3')

    print(f"\n  tech_3 (gerçek sinyal) ΔPR-AUC = {signal_drop:+.4f}")
    print(f"  diğer 9 özellik        maks    = {others.abs().max():+.4f}")

    ok1 = signal_drop > 0.20
    ok2 = others.abs().max() < 0.02
    grp = imp[imp['kind'] == 'group'].set_index('target')['drop_pr_auc']
    ok3 = grp['GRUP: teknik (I)'] > 0.20 and abs(grp['GRUP: fundamental (II)']) < 0.02

    for name, ok in [('sinyal özelliği tespit edildi', ok1),
                     ('gürültü özellikleri ≈ 0', ok2),
                     ('grup düzeyi doğru ayrıştı', ok3)]:
        print(f"  [{'GEÇTİ' if ok else 'KALDI'}] {name}")

    if ok1 and ok2 and ok3:
        print("\n  ✓ Permütasyon önemi kodu doğru çalışıyor.")
        return True
    print("\n  ✗ TEST BAŞARISIZ — attribution çıktılarını kullanmayın.")
    return False


# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--channels', nargs='+',
                    default=['permutation', 'attention', 'gate', 'film'],
                    choices=['permutation', 'attention', 'gate', 'film'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--groups-only', action='store_true')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)

    device = get_device()
    print("═" * 78)
    print("TRANSFORMER ATTRIBUTION")
    print("═" * 78)
    print(f"  Kanallar : {args.channels}")
    print(f"  Seed'ler : {args.seeds}")
    print(f"  Device   : {device}")
    print(f"  Eşik     : her seed için YALNIZCA validation'dan seçilir\n")

    dataset_out = prepare_dataset(force_refresh=False)

    # Tüm kanallar aynı I+II özellik kümesini kullanır → tek loader yeter
    tech_groups, fund_groups, _, _ = ABLATIONS['multi_pure']
    _, val_loader, test_loader, _, (tech_cols, fund_cols) = get_dual_stream_dataloaders(
        dataset_out, seq_len=BASE_CONFIG['seq_len'], batch_size=args.batch_size,
        tech_groups=tech_groups, fund_groups=fund_groups,
    )
    val_ds, test_ds = val_loader.dataset, test_loader.dataset
    print(f"\n  tech_dim={len(tech_cols)} | fund_dim={len(fund_cols)} | "
          f"test n={len(test_ds):,}")

    # Rejim bilgisi (tarih → VIX, yıl)
    macro_cols = [c for c in ('VIX_Close', 'Is_Yield_Inverted') if c in dataset_out.columns]
    macro_by_date = dataset_out.groupby(dataset_out.index)[macro_cols].first()

    perm_all, gate_frames, attn_all, film_all, gate_stats = [], [], [], [], []

    for ablation in sorted({CHANNEL_ABLATION[c] for c in args.channels}):
        chans = [c for c in args.channels if CHANNEL_ABLATION[c] == ablation]
        print("\n" + "▄" * 78)
        print(f"CHECKPOINT AİLESİ: {ablation}  (fusion={fusion_for(ablation)})")
        print(f"  Besleyeceği kanallar: {chans}")
        print("▄" * 78)

        available = [s for s in args.seeds if os.path.exists(ckpt_path(ablation, s))]
        if not available:
            print(f"  ⚠️ Hiç checkpoint bulunamadı ({ckpt_path(ablation, args.seeds[0])})")
            print(f"     Önce şunu koşun:")
            print(f"       python run_modality_v2.py --ablations {ablation}")
            print(f"     Bu kanallar atlanıyor: {chans}")
            continue
        if len(available) < len(args.seeds):
            print(f"  ⚠️ {len(available)}/{len(args.seeds)} seed mevcut: {available}")

        for seed in available:
            print(f"\n  ── seed {seed} ──")
            model = build_model(ablation, len(tech_cols), len(fund_cols))
            model.load_state_dict(torch.load(ckpt_path(ablation, seed),
                                             map_location=device, weights_only=True))
            model.to(device).eval()

            # Eşik: yalnızca validation'dan
            val_prob = predict(model, val_ds, device, args.batch_size)
            thr, val_mcc = find_best_threshold_mcc(val_ds.labels, val_prob)
            print(f"    val eşiği={thr:.4f} (val MCC={val_mcc:.4f})")

            if 'permutation' in chans:
                imp = permutation_importance(
                    model, test_ds, device, thr, tech_cols, fund_cols,
                    repeats=args.repeats, groups_only=args.groups_only,
                    rng_seed=seed, batch_size=args.batch_size,
                )
                imp['seed'] = seed
                perm_all.append(imp)

            need_internals = any(c in chans for c in ('gate', 'attention', 'film'))
            if need_internals:
                cap = capture_internals(model, test_ds, device, args.batch_size)

                if 'gate' in chans and 'gate_t' in cap:
                    g = pd.DataFrame({
                        'date': test_ds.dates.values, 'ticker': test_ds.tickers,
                        'seed': seed,
                        'gate_t_tech_uses_fund': cap['gate_t'],
                        'gate_f_fund_uses_tech': cap['gate_f'],
                    })
                    g['year'] = pd.to_datetime(g['date']).dt.year
                    g = g.merge(macro_by_date, left_on='date', right_index=True, how='left')
                    gate_frames.append(g)
                    print(f"    kapı ort: tech←fund={cap['gate_t'].mean():.4f} | "
                          f"fund←tech={cap['gate_f'].mean():.4f}")

                if 'attention' in chans and 'attn_t2f' in cap:
                    attn_all.append({'seed': seed, 'profile': cap['attn_t2f'].mean(axis=0)})
                    print(f"    attention profili yakalandı ({cap['attn_t2f'].shape[1]} token)")

                if 'film' in chans and 'film_gamma' in cap:
                    film_all.append({'seed': seed,
                                     'gamma_mean': float(cap['film_gamma'].mean()),
                                     'gamma_std': float(cap['film_gamma'].std())})
                    print(f"    FiLM gamma ort={cap['film_gamma'].mean():.4f}")

    # ══════════════════════════════════════════════════════════════
    # RAPORLAMA
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 78)
    print("SONUÇLAR")
    print("═" * 78)

    # ── [1] Permütasyon ──
    if perm_all:
        P = pd.concat(perm_all, ignore_index=True)
        P.to_csv(os.path.join(OUT_CSV, 'attr_permutation_raw.csv'), index=False)
        agg = (P.groupby(['target', 'kind', 'stream'])
                 .agg(drop_pr_auc=('drop_pr_auc', 'mean'),
                      drop_pr_auc_sd=('drop_pr_auc', 'std'),
                      drop_mcc=('drop_mcc', 'mean'),
                      n=('seed', 'nunique'))
                 .reset_index()
                 .sort_values('drop_pr_auc', ascending=False))
        agg.to_csv(os.path.join(OUT_CSV, 'attr_permutation_summary.csv'), index=False)

        print("\n── [1] PERMÜTASYON ÖNEMİ (grup düzeyi) ──")
        print(f"{'Grup':<28}{'ΔPR-AUC':>11}{'±sd':>9}{'ΔMCC':>10}{'n':>4}")
        print("-" * 62)
        for _, r in agg[agg['kind'] == 'group'].iterrows():
            print(f"{r['target']:<28}{r['drop_pr_auc']:>+11.5f}"
                  f"{(r['drop_pr_auc_sd'] if pd.notna(r['drop_pr_auc_sd']) else 0):>9.5f}"
                  f"{r['drop_mcc']:>+10.5f}{int(r['n']):>4}")

        feat = agg[agg['kind'] == 'feature']
        if not feat.empty:
            print("\n── En önemli 10 özellik ──")
            print(f"{'Özellik':<28}{'akış':>8}{'ΔPR-AUC':>11}")
            print("-" * 47)
            for _, r in feat.head(10).iterrows():
                print(f"{r['target']:<28}{r['stream']:>8}{r['drop_pr_auc']:>+11.5f}")

            top = feat.head(20).iloc[::-1]
            fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.32)))
            ax.barh(range(len(top)), top['drop_pr_auc'],
                    xerr=top['drop_pr_auc_sd'].fillna(0),
                    color=[C[s] for s in top['stream']], error_kw=dict(lw=0.8, alpha=0.6))
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top['target'], fontsize=8)
            ax.axvline(0, color=C['gray'], lw=1)
            ax.set_xlabel('PR-AUC düşüşü (permütasyon önemi)')
            ax.set_title('Transformer permütasyon önemi — en önemli 20 özellik\n'
                         f"mavi = teknik (I), bordo = fundamental (II) · "
                         f"{agg['n'].max()} seed ortalaması",
                         fontsize=11, loc='left')
            ax.grid(alpha=0.25, axis='x')
            save_fig(fig, 'fig_attr1_permutation_importance.png')

        grp = agg[agg['kind'] == 'group']
        if len(grp) == 2:
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            ax.bar(grp['target'], grp['drop_pr_auc'],
                   yerr=grp['drop_pr_auc_sd'].fillna(0),
                   color=[C[s] for s in grp['stream']], width=0.55,
                   error_kw=dict(lw=1, alpha=0.7))
            ax.axhline(0, color=C['gray'], lw=1)
            ax.set_ylabel('PR-AUC düşüşü')
            ax.set_title('Modalite grubu permütasyon önemi (I+II modeli)\n'
                         'Ablation bulgusunun bağımsız doğrulaması',
                         fontsize=11, loc='left')
            ax.grid(alpha=0.25, axis='y')
            save_fig(fig, 'fig_attr2_group_importance.png')

    # ── [2] Kapı ──
    if gate_frames:
        G = pd.concat(gate_frames, ignore_index=True)
        G.to_csv(os.path.join(OUT_CSV, 'attr_gate_raw.csv'), index=False)
        gcols = ['gate_t_tech_uses_fund', 'gate_f_fund_uses_tech']
        # Seed başına ayrı analiz — groupby.apply yerine açık döngü
        # (pandas sürümleri arasında apply'ın index davranışı değişiyor)
        parts = []
        for s, sub in G.groupby('seed'):
            r = gate_regime_analysis(sub, gcols)
            r.insert(0, 'seed', s)
            parts.append(r)
        stats_df = pd.concat(parts, ignore_index=True)
        stats_df.to_csv(os.path.join(OUT_CSV, 'attr_gate_regime.csv'), index=False)

        print("\n── [2] KAPI DEĞERLERİ (I+II, makrosuz) ──")
        print(f"{'Kapı':<26}{'ortalama':>10}{'±sd(seed)':>12}")
        print("-" * 48)
        for c in gcols:
            per_seed = G.groupby('seed')[c].mean()
            print(f"{c:<26}{per_seed.mean():>10.4f}{per_seed.std():>12.4f}")
        print("  Yorum: 0.5 = nötr. <0.5 → karşı modalite bastırılıyor.")

        print(f"\n  VIX ile ilişki (seed başına Spearman ρ):")
        vix_rows = stats_df[(stats_df['vs'] == 'VIX_Close')]
        for c in gcols:
            sub = vix_rows[vix_rows['gate'] == c]
            if sub.empty:
                continue
            sig = int((sub['p'] < 0.05).sum())
            print(f"    {c:<26} ρ={sub['value'].mean():+.4f} ± {sub['value'].std():.4f}"
                  f" | p<0.05 olan seed: {sig}/{len(sub)}")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for ax, c in zip(axes, gcols):
            by_year = G.groupby(['seed', 'year'])[c].mean().unstack()
            for s in by_year.index:
                ax.plot(by_year.columns, by_year.loc[s], marker='o', alpha=0.55,
                        lw=1.2, label=f'seed {s}')
            ax.axhline(0.5, color=C['gray'], ls='--', lw=1)
            ax.set_title(c, fontsize=10, loc='left')
            ax.set_xlabel('Yıl'); ax.set_ylabel('ortalama kapı değeri')
            ax.set_xticks(sorted(G['year'].unique()))
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=7)
        fig.suptitle('Kapı değerleri rejime göre — kesikli çizgi nötr (0.5)',
                     fontsize=11, x=0.09, ha='left')
        save_fig(fig, 'fig_attr3_gate_by_regime.png')

    # ── [3] Attention profili ──
    if attn_all:
        prof = np.vstack([a['profile'] for a in attn_all])   # (n_seed, T+1)
        m, sd = prof.mean(0), prof.std(0)
        pd.DataFrame({'token': np.arange(len(m)), 'mean': m, 'std': sd}).to_csv(
            os.path.join(OUT_CSV, 'attr_attention_profile.csv'), index=False)

        uniform = 1.0 / len(m)
        print("\n── [3] CROSS-ATTENTION PROFİLİ (tech CLS → fund token'ları) ──")
        print(f"  Uniform taban   : {uniform:.4f}")
        print(f"  CLS token'a     : {m[0]:.4f}")
        print(f"  En yüksek gün   : t-{len(m) - 1 - int(np.argmax(m[1:])) - 1} "
              f"({m[1:].max():.4f})")
        conc = float(np.abs(m - uniform).sum())
        print(f"  Uniform'dan sapma (L1): {conc:.4f}  "
              f"({'yoğunlaşmış' if conc > 0.2 else 'neredeyse düz'})")

        fig, ax = plt.subplots(figsize=(10, 3.8))
        x = np.arange(len(m))
        ax.bar(x, m, yerr=sd, color=C['tech'], error_kw=dict(lw=0.8, alpha=0.6))
        ax.axhline(uniform, color=C['accent'], ls='--', lw=1.2,
                   label=f'uniform ({uniform:.3f})')
        ax.set_xticks(x)
        ax.set_xticklabels(['CLS'] + [f't-{len(m) - 1 - i}' for i in range(1, len(m))],
                           fontsize=7, rotation=45)
        ax.set_ylabel('ortalama dikkat ağırlığı')
        ax.set_title('Cross-attention: teknik CLS token\'ı fundamental akışın '
                     'hangi zaman adımına bakıyor?\n'
                     'Uniform çizgisine yakınlık = cross-attention fiilen '
                     'ayrım yapmıyor', fontsize=11, loc='left')
        ax.legend(fontsize=8); ax.grid(alpha=0.25, axis='y')
        save_fig(fig, 'fig_attr4_attention_profile.png')

    # ── [4] FiLM ──
    if film_all:
        F = pd.DataFrame(film_all)
        F.to_csv(os.path.join(OUT_CSV, 'attr_film_gamma.csv'), index=False)
        print("\n── [4] FiLM GAMMA (I+II, makrosuz) ──")
        print(f"  gamma = {F['gamma_mean'].mean():.4f} ± {F['gamma_mean'].std():.4f} "
              f"({len(F)} seed)")
        print("  Yorum: 1.0 = kimlik. <1 → fundamental bağlam technical sinyali "
              "bastırıyor.")

    print("\n" + "═" * 78)
    print("Çıktılar → results/attr_*.csv  ve  visualization/attribution/*.png")
    print("═" * 78)


if __name__ == '__main__':
    main()