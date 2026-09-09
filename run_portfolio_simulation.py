"""
run_portfolio_simulation.py
────────────────────────────
Tezin FİNAL ekonomik değerlendirmesi: portföy simülasyonu.

═══════════════════════════════════════════════════════════════════════
TASARIM KARARLARI (ve gerekçeleri)
═══════════════════════════════════════════════════════════════════════
1. SADECE UZUN YÖN. Tezin iddiası "drawdown riskini görüp çıkmak" — açığa
   satış değil. Ayrıca small cap'lerde açığa satış varsayımı tartışmalıdır.

2. 20 GÜNDE BİR YENİDEN DENGELEME. Tahmin ufkunuz 20 gün; günlük dengeleme
   hem gerçekçi değil hem turnover'ı yapay şişiriyor (önceki backtest'te
   sonucu bulanıklaştıran etkenlerden biriydi).

3. SABİT BÜTÇE (en riskli %k çıkarılır), olasılık eşiği DEĞİL. Eşik rejimler
   arası kayıyordu (flag rate %22/%35/%33). Sabit bütçe rejimden bağımsız ve
   a priori tanımlı — kalibrasyon kayması sorunu ortadan kalkıyor.

4. BİRİNCİL METRİK: MAKSİMUM DRAWDOWN ve CALMAR. Sharpe değil.
   İddianız aşağı yönlü koruma; Sharpe yukarı oynaklığı da cezalandırır.
   Bir de-risking stratejisinin boğa piyasasında getirisi buy&hold'un
   ALTINDA olması normaldir — soru, drawdown'ın yeterince iyileşip
   iyileşmediğidir.

═══════════════════════════════════════════════════════════════════════
KARŞILAŞTIRILANLAR
═══════════════════════════════════════════════════════════════════════
  buy_hold     Tüm evren, eşit ağırlık. KARŞI-OLGUSAL: "hiçbir şey
               yapmasaydınız ne olurdu?" Bu olmadan hiçbir getiri sayısının
               anlamı yoktur.
  naive_vol    Vol_20d'ye göre en oynak %k çıkarılır. UCUZ ALTERNATİF:
               model, "oynaklardan kaç" kuralını geçiyor mu? Geçmiyorsa
               51 özellikli derin modelin ekonomik katkısı yoktur.
  model_*      Model olasılığına göre en riskli %k çıkarılır.
  oracle       GERÇEK etiketlere göre en riskli %k çıkarılır — mükemmel
               öngörü. TAVAN: bu etiket ve bu strateji ile en fazla ne
               kazanılabilir?
                 · Oracle drawdown'ı belirgin düşürüyorsa → strateji sağlam,
                   sınır modelin tahmin gücünde.
                 · Oracle bile düşürmüyorsa → STRATEJİ TASARIMI hatalı,
                   hiçbir model bu kurulumla değer üretemez.
               Bu ayrım olmadan "ekonomik değer yok" demek eksiktir.
               (Oracle bilinçli olarak geleceğe bakar; bir strateji önerisi
                değil, üst sınır referansıdır.)

ZAMANLAMA: sinyal t gününde üretilir, pozisyon t+1'den bir sonraki
rebalance gününe kadar tutulur. İleriye bakış yok (oracle hariç, o tanımı
gereği bakar).

KULLANIM:
    python run_portfolio_simulation.py
    python run_portfolio_simulation.py --exclude-pct 10 --rebalance 20
    python run_portfolio_simulation.py --no-xgb
"""
import os
import argparse

import numpy as np
import pandas as pd
import torch

from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer
from models.pytorch_trainer import _evaluate_on_loader
from run_modality_v2 import ABLATIONS, BASE_CONFIG

TEST_START, TEST_END = '2022-01-01', '2024-12-31'
TRADING_DAYS = 252


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ══════════════════════════════════════════════════════════════════
# Performans metrikleri
# ══════════════════════════════════════════════════════════════════
def performance(ret: pd.Series, label=''):
    """
    Günlük net getiri serisinden performans metrikleri.

    ÖNEMLİ: anahtar seti HER DURUMDA aynıdır. Kısa/boş seride boş sözlük
    döndürmek, sonraki tablolarda o kolonun hiç oluşmamasına ve KeyError'a
    yol açıyordu. Artık eksik değerler NaN olarak döner — satır düşer ama
    şema bozulmaz.
    """
    keys = ['strategy', 'total_return', 'cagr', 'ann_vol', 'sharpe',
            'sortino', 'max_drawdown', 'calmar', 'n_days']
    r = ret.dropna()
    if len(r) < 20:
        out = {k: np.nan for k in keys}
        out['strategy'] = label
        out['n_days'] = len(r)
        return out
    cum = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    total = float(cum.iloc[-1] - 1)
    cagr = float(cum.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else 0.0
    downside = r[r < 0]
    sortino = (float(r.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS))
               if len(downside) > 1 and downside.std(ddof=1) > 0 else np.nan)
    run_max = np.maximum.accumulate(cum.values)
    maxdd = float(((cum.values - run_max) / run_max).min())
    calmar = float(cagr / abs(maxdd)) if maxdd < 0 else np.nan
    return {'strategy': label, 'total_return': total, 'cagr': cagr, 'ann_vol': vol,
            'sharpe': sharpe, 'sortino': sortino, 'max_drawdown': maxdd,
            'calmar': calmar, 'n_days': len(r)}


def simulate(scores_wide, ret_wide, rebal_dates, exclude_pct, cost_bps,
             higher_is_riskier=True):
    """
    Sabit bütçeli de-risking simülasyonu.

    scores_wide : (tarih × hisse) risk skoru. NaN = veri yok, portföye alınmaz.
    ret_wide    : (tarih × hisse) günlük getiri
    rebal_dates : yeniden dengeleme günleri
    exclude_pct : her rebalance'ta çıkarılacak en riskli yüzde
    Ağırlıklar rebalance gününde belirlenir, ERTESİ GÜNDEN itibaren uygulanır.
    """
    weights = pd.DataFrame(0.0, index=ret_wide.index, columns=ret_wide.columns)
    cur = pd.Series(0.0, index=ret_wide.columns)
    turnover_log = {}

    for d in ret_wide.index:
        if d in rebal_dates:
            s = scores_wide.loc[d] if d in scores_wide.index else pd.Series(dtype=float)
            avail = s.dropna()
            # O gün getirisi olan hisselerle sınırla
            avail = avail[avail.index.isin(ret_wide.columns)]
            if len(avail) >= 10:
                k = int(np.ceil(len(avail) * exclude_pct / 100.0))
                # ── SIRALAMA YÖNÜ ──
                # higher_is_riskier=True ise ARTAN sıralama gerekir:
                # [en güvenli ... en riskli]. keep = ilk (n−k) → en güvenliler
                # tutulur, son k (en riskliler) çıkarılır.
                # (Önceki sürümde 'ascending=not higher_is_riskier' yazılıydı;
                #  bu, listeyi en riskliden başlatıp EN RİSKLİLERİ TUTUYORDU —
                #  stratejiyi tam tersine çeviren bir hataydı.)
                # kind='mergesort' → kararlı sıralama. Beraberlik durumunda
                # (ör. aynı skoru alan hisseler) sonuç keyfi olmasın,
                # tekrarlanabilir kalsın. Gerçek olasılıklarda beraberlik
                # pratikte olmaz ama determinizm ucuz bir sigorta.
                ranked = avail.sort_values(ascending=higher_is_riskier,
                                           kind='mergesort')
                keep = ranked.index[:len(avail) - k] if k > 0 else ranked.index

                # Koruma: tutulanların ortalama risk skoru, çıkarılanlardan
                # DÜŞÜK olmalı. Değilse yön yine ters demektir — sessizce
                # geçmesin, hemen patlasın.
                if k > 0 and len(keep) > 0:
                    dropped = ranked.index[len(avail) - k:]
                    if avail[keep].mean() > avail[dropped].mean():
                        raise RuntimeError(
                            f"Sıralama yönü ters! Tutulanların ort. risk skoru "
                            f"({avail[keep].mean():.4f}) çıkarılanlardan "
                            f"({avail[dropped].mean():.4f}) yüksek.")

                new = pd.Series(0.0, index=ret_wide.columns)
                if len(keep) > 0:
                    new[keep] = 1.0 / len(keep)
                turnover_log[d] = float((new - cur).abs().sum() / 2.0)
                cur = new
        weights.loc[d] = cur

    # Sinyal t'de, getiri t+1'de → shift(1)
    gross = (weights.shift(1).fillna(0.0) * ret_wide).sum(axis=1)
    cost = pd.Series(0.0, index=ret_wide.index)
    for d, t in turnover_log.items():
        cost.loc[d] = t * (cost_bps / 10_000.0)
    net = gross - cost
    avg_turnover = float(np.mean(list(turnover_log.values()))) if turnover_log else 0.0
    exposure = float((weights.sum(axis=1) > 0).mean())
    return net, avg_turnover, exposure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablation', type=str, default='multi_pure', choices=list(ABLATIONS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--exclude-pct', type=float, default=20.0,
                    help='Her rebalance\'ta çıkarılacak en riskli yüzde')
    ap.add_argument('--rebalance', type=int, default=20,
                    help='Yeniden dengeleme aralığı (işlem günü)')
    ap.add_argument('--cost-bps', type=float, nargs='+', default=[10, 25, 50])
    ap.add_argument('--no-xgb', action='store_true')
    ap.add_argument('--ckpt-dir', type=str, default='checkpoints')
    ap.add_argument('--outdir', type=str, default='results')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()

    print("═" * 78)
    print("PORTFÖY SİMÜLASYONU — final ekonomik değerlendirme")
    print("═" * 78)
    print(f"  Model        : {args.ablation} ({len(args.seeds)} seed)")
    print(f"  Çıkarma      : en riskli %{args.exclude_pct:.0f} (sabit bütçe)")
    print(f"  Rebalance    : {args.rebalance} işlem günü")
    print(f"  Maliyet      : {args.cost_bps} bps")
    print(f"  Birincil ölçüt: MAKS. DRAWDOWN ve CALMAR (Sharpe değil)")
    print()

    dataset_out = prepare_dataset(force_refresh=False)
    tech_groups, fund_groups, modality, _ = ABLATIONS[args.ablation]
    _, val_loader, test_loader, _, (tech_cols, fund_cols) = get_dual_stream_dataloaders(
        dataset_out, seq_len=BASE_CONFIG['seq_len'], batch_size=64,
        tech_groups=tech_groups, fund_groups=fund_groups)
    ds = test_loader.dataset

    # ── Fiyat / getiri matrisi ──
    m = (dataset_out.index >= TEST_START) & (dataset_out.index <= TEST_END)
    sub = dataset_out[m]
    price = sub.pivot_table(index=sub.index, columns='Ticker', values='Close')
    ret_wide = price.pct_change().fillna(0.0)

    # Rebalance günleri
    all_days = ret_wide.index.sort_values()
    rebal = set(all_days[::args.rebalance])
    print(f"  Test dönemi  : {len(all_days)} gün | {ret_wide.shape[1]} hisse "
          f"| {len(rebal)} rebalance\n")

    # ── Skor matrisleri ──
    base = pd.DataFrame({'date': ds.dates.values, 'ticker': ds.tickers,
                         'target': ds.labels})

    def to_wide(values):
        d = base.copy(); d['v'] = values
        w = d.pivot_table(index='date', columns='ticker', values='v')
        return w.reindex(index=all_days).reindex(columns=ret_wide.columns)

    scores = {}

    # 1) Model — seed başına
    for seed in args.seeds:
        ck = os.path.join(args.ckpt_dir, f'best_v2_{args.ablation}_seed{seed}.pth')
        if not os.path.exists(ck):
            print(f"  ⚠️ yok: {ck}")
            continue
        mdl = DualEncoderTransformer(
            tech_dim=len(tech_cols), fund_dim=len(fund_cols),
            seq_len=BASE_CONFIG['seq_len'], d_model=BASE_CONFIG['d_model'],
            n_heads=BASE_CONFIG['n_heads'], n_layers=BASE_CONFIG['n_layers'],
            dropout=BASE_CONFIG['dropout'], modality=modality,
            fusion_type=BASE_CONFIG['fusion_type'])
        mdl.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        mdl.to(device)
        p, _, _ = _evaluate_on_loader(mdl, test_loader, device)
        scores[f'model_seed{seed}'] = to_wide(p)
        del mdl
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f"  ✓ model seed={seed} tahminleri alındı")

    # 2) Naif volatilite
    if 'Vol_20d' in sub.columns:
        scores['naive_vol'] = (sub.pivot_table(index=sub.index, columns='Ticker',
                                               values='Vol_20d')
                               .reindex(index=all_days).reindex(columns=ret_wide.columns))
        print("  ✓ naif volatilite skoru hazır")

    # 3) Oracle — gerçek etiket (bilinçli ileriye bakış, ÜST SINIR referansı)
    scores['oracle'] = to_wide(base['target'].values)
    print("  ✓ oracle (mükemmel öngörü) hazır")

    # 4) XGBoost — kayıtlı parametrelerle hızlı yeniden eğitim
    if not args.no_xgb:
        pp = os.path.join(args.outdir, 'xgb_factorial_params.csv')
        if os.path.exists(pp):
            try:
                import xgboost as xgb
                from sklearn.preprocessing import RobustScaler
                from datasets.feature_engineering import resolve_groups
                pdf = pd.read_csv(pp)
                row = pdf[pdf['cell'] == 'I+II+IV']
                if not row.empty:
                    cols = resolve_groups(dataset_out, ('tech', 'fund'))
                    tr = dataset_out[dataset_out.index <= '2019-12-31']
                    va = dataset_out[(dataset_out.index >= '2020-01-01') &
                                     (dataset_out.index <= '2021-12-31')]
                    med = tr[cols].median()
                    sc = RobustScaler()
                    Xtr = sc.fit_transform(tr[cols].fillna(med).values)
                    Xva = sc.transform(va[cols].fillna(med).values)
                    Xte = sc.transform(sub[cols].fillna(med).values)
                    prm = {k: v for k, v in row.iloc[0].items()
                           if k not in ('cell', 'n_features', 'cv_pr_auc') and pd.notna(v)}
                    prm['tree_method'] = 'hist'
                    mx = xgb.XGBClassifier(**prm, eval_metric='logloss', random_state=42,
                                           n_jobs=-1, early_stopping_rounds=50)
                    mx.fit(Xtr, tr['Target'].values,
                           eval_set=[(Xva, va['Target'].values)], verbose=False)
                    pte = mx.predict_proba(Xte)[:, 1]
                    # .values ŞART: sub.index adı 'date' ve sub['Ticker'] aynı
                    # indeksi taşıyor. Doğrudan verilirse DataFrame'de hem
                    # 'date' indeksi hem 'date' kolonu oluşur ve pivot_table
                    # "ambiguous" hatası verir. .values indeks bağını koparır.
                    xw = pd.DataFrame({'date': sub.index.values,
                                       'ticker': sub['Ticker'].values,
                                       'v': pte}).pivot_table(
                        index='date', columns='ticker', values='v')
                    scores['xgboost'] = xw.reindex(index=all_days).reindex(
                        columns=ret_wide.columns)
                    print("  ✓ XGBoost (I+II) tahminleri hazır")
            except Exception as e:
                print(f"  ⚠️ XGBoost atlandı: {e}")
        else:
            print(f"  ⚠️ {pp} yok — XGBoost atlandı "
                  f"(önce run_xgb_factorial.py çalıştırın)")

    # ══ ETİKET–EKONOMİ TEŞHİSİ ══
    # En kritik soru: Target=1 (kriz) etiketi, GERÇEKTEN kayıp anlamına
    # geliyor mu? Bir hisse ay içinde sert düşüp ayı yükselişte kapatabilir;
    # o durumda "drawdown yaşadı" doğru ama "para kaybettirdi" yanlıştır.
    # Oracle'ın buy&hold'un altında kalması bu hizasızlığın işaretiydi —
    # burada doğrudan ölçüyoruz.
    print("\n" + "═" * 78)
    print("ETİKET–EKONOMİ HİZALAMASI")
    print("═" * 78)
    fwd = (price.shift(-args.rebalance) / price - 1)      # ileriye dönük getiri
    tgt_w = sub.pivot_table(index=sub.index, columns='Ticker', values='Target')
    idx = fwd.index.intersection(tgt_w.index)
    F = fwd.loc[idx].stack(dropna=True)
    T = tgt_w.loc[idx].stack(dropna=True)
    j = pd.DataFrame({'fwd': F, 'tgt': T}).dropna()
    j['year'] = j.index.get_level_values(0).year

    lab_rows = []
    print(f"{'dönem':<8}{'n':>10}{'kriz%':>8}"
          f"{'ort.getiri(T=1)':>17}{'ort.getiri(T=0)':>17}{'fark':>9}{'medyan fark':>13}")
    print("-" * 82)
    for yr in [2022, 2023, 2024, 'TÜM']:
        s = j if yr == 'TÜM' else j[j['year'] == yr]
        if len(s) < 100:
            continue
        a, b = s[s.tgt == 1]['fwd'], s[s.tgt == 0]['fwd']
        d_mean = a.mean() - b.mean()
        d_med = a.median() - b.median()
        lab_rows.append({'period': yr, 'n': len(s), 'crisis_rate': s.tgt.mean(),
                         'fwd_ret_target1': a.mean(), 'fwd_ret_target0': b.mean(),
                         'diff_mean': d_mean, 'diff_median': d_med})
        print(f"{str(yr):<8}{len(s):>10,}{s.tgt.mean()*100:>7.1f}%"
              f"{a.mean()*100:>16.2f}%{b.mean()*100:>16.2f}%"
              f"{d_mean*100:>+8.2f}%{d_med*100:>+12.2f}%")
    pd.DataFrame(lab_rows).to_csv(
        os.path.join(args.outdir, 'label_economics.csv'), index=False)

    print("\n  Yorum:")
    all_row = [r for r in lab_rows if r['period'] == 'TÜM']
    if all_row:
        dm = all_row[0]['diff_mean']
        if dm > 0:
            print(f"  → Target=1 hisselerinin ileriye dönük {args.rebalance} günlük")
            print(f"    ortalama getirisi, Target=0'dan {dm*100:+.2f} puan DAHA YÜKSEK.")
            print("    Etiket ekonomik hedefle HİZALI DEĞİL: 'düşüş yaşayacak' ile")
            print("    'para kaybettirecek' aynı şey değil. Toparlanan düşüş kayıp")
            print("    değildir — ve en yüksek getirili hisseler tam olarak bunu yapar.")
            print("    Bu, oracle'ın neden buy&hold'un altında kaldığını açıklar.")
        else:
            print(f"  → Target=1 hisseleri {abs(dm)*100:.2f} puan DAHA DÜŞÜK getiri")
            print("    sağlıyor: etiket ekonomik hedefle hizalı. Oracle'ın zayıflığı")
            print("    başka bir kaynaktan geliyor (strateji tasarımı / rejim).")

    # ══ Simülasyon ══
    rows, curves = [], {}

    def record(net, name, cost, tov, exp_):
        """Tüm dönem + yıl bazında performans kaydeder."""
        r = performance(net, name)
        r.update({'cost_bps': cost, 'avg_turnover': tov, 'exposure': exp_,
                  'period': 'ALL'})
        rows.append(r)
        for yr in sorted(set(net.index.year)):
            sy = net[net.index.year == yr]
            ry = performance(sy, name)
            ry.update({'cost_bps': cost, 'avg_turnover': tov,
                       'exposure': exp_, 'period': str(yr)})
            rows.append(ry)

    for cost in args.cost_bps:
        # Buy & hold (karşı-olgusal)
        bh_scores = pd.DataFrame(0.0, index=all_days, columns=ret_wide.columns)
        net, tov, exp_ = simulate(bh_scores, ret_wide, rebal, 0.0, cost)
        record(net, 'buy_hold', cost, tov, exp_)
        curves[f'buy_hold_{cost}'] = (1 + net).cumprod()

        for name, sw in scores.items():
            net, tov, exp_ = simulate(sw, ret_wide, rebal, args.exclude_pct, cost)
            record(net, name, cost, tov, exp_)
            curves[f'{name}_{cost}'] = (1 + net).cumprod()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, 'portfolio_simulation.csv'), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(args.outdir, 'portfolio_curves.csv'))

    # ── Yıl bazında tablo ──
    dfy = df.copy()
    dfy['grp'] = dfy['strategy'].str.replace(r'model_seed\d+', 'model', regex=True)
    c0 = args.cost_bps[0]
    print("\n" + "═" * 78)
    print(f"YIL BAZINDA (maliyet {c0:.0f} bps)")
    print("═" * 78)
    for per in ['2022', '2023', '2024']:
        s = dfy[(dfy['period'] == per) & (dfy['cost_bps'] == c0)]
        if s.empty:
            continue
        gg = s.groupby('grp').agg(ret=('total_return', 'mean'),
                                  mdd=('max_drawdown', 'mean'),
                                  cal=('calmar', 'mean'),
                                  srt=('sortino', 'mean'))
        print(f"\n── {per} ──")
        print(f"{'strateji':<12}{'getiri':>10}{'maxDD':>10}{'Calmar':>9}{'Sortino':>9}")
        for o in ['buy_hold', 'naive_vol', 'model', 'xgboost', 'oracle']:
            if o in gg.index:
                r = gg.loc[o]
                print(f"{o:<12}{r.ret*100:>9.1f}%{r.mdd*100:>9.1f}%"
                      f"{r.cal:>9.2f}{r.srt:>9.2f}")
        if {'oracle', 'buy_hold'} <= set(gg.index):
            # NOT: yukarıdaki agg() kolonu 'cal' olarak adlandırıyor, 'calmar' değil.
            d = gg.loc['oracle', 'cal'] - gg.loc['buy_hold', 'cal']
            verdict = ("oracle ÜSTÜN → etiket bu rejimde işe yarıyor" if d > 0
                       else "oracle ZAYIF → etiket bu rejimde de hizalı değil")
            print(f"  oracle − buy&hold (Calmar): {d:+.2f}   {verdict}")

    # ── Model seed'lerini birleştir ──
    # ÖNEMLİ: df artık hem 'ALL' hem yıl bazlı satırlar içeriyor.
    # Tüm-dönem özetinde SADECE period=='ALL' satırları kullanılmalı,
    # aksi halde yıl satırları ortalamaya karışır ve sayılar bozulur.
    def agg(d):
        d = d[d['period'] == 'ALL'].copy()
        d['grp'] = d['strategy'].str.replace(r'model_seed\d+', 'model', regex=True)
        return d.groupby(['grp', 'cost_bps']).agg(
            total_return=('total_return', 'mean'), cagr=('cagr', 'mean'),
            sharpe=('sharpe', 'mean'), sortino=('sortino', 'mean'),
            max_drawdown=('max_drawdown', 'mean'), maxdd_sd=('max_drawdown', 'std'),
            calmar=('calmar', 'mean'), turnover=('avg_turnover', 'mean')).reset_index()

    g = agg(df)
    order = ['buy_hold', 'naive_vol', 'model', 'xgboost', 'oracle']

    print("\n" + "═" * 78)
    print(f"SONUÇLAR — test 2022-2024, en riskli %{args.exclude_pct:.0f} çıkarılıyor")
    print("═" * 78)
    for cost in args.cost_bps:
        print(f"\n── işlem maliyeti {cost:.0f} bps ──")
        print(f"{'strateji':<12}{'toplam':>10}{'CAGR':>9}{'maxDD':>10}{'Calmar':>9}"
              f"{'Sharpe':>9}{'Sortino':>9}{'turnover':>10}")
        print("-" * 78)
        sub_g = g[g['cost_bps'] == cost]
        for s in order:
            r = sub_g[sub_g['grp'] == s]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"{s:<12}{r.total_return*100:>9.1f}%{r.cagr*100:>8.1f}%"
                  f"{r.max_drawdown*100:>9.1f}%{r.calmar:>9.2f}{r.sharpe:>9.2f}"
                  f"{r.sortino:>9.2f}{r.turnover*100:>9.1f}%")

    # ── Yorum ──
    print("\n" + "═" * 78)
    print("YORUM")
    print("═" * 78)
    c0 = args.cost_bps[0]
    gg = g[g['cost_bps'] == c0].set_index('grp')
    if {'buy_hold', 'oracle'} <= set(gg.index):
        bh, orc = gg.loc['buy_hold'], gg.loc['oracle']
        d_orc = orc['max_drawdown'] - bh['max_drawdown']
        print(f"  Oracle maxDD iyileştirmesi : {d_orc*100:+.2f} puan "
              f"({bh['max_drawdown']*100:.1f}% → {orc['max_drawdown']*100:.1f}%)")
        if d_orc < 0.02:
            print("  → MÜKEMMEL öngörü bile drawdown'ı anlamlı düşürmüyor.")
            print("    Sınır modelde değil, STRATEJİ TASARIMINDA. Hiçbir tahmin")
            print("    modeli bu kurulumla ekonomik değer üretemez — bu, tezde")
            print("    raporlanması gereken güçlü ve kesin bir sonuçtur.")
        else:
            print("  → Mükemmel öngörü drawdown'ı düşürebiliyor: strateji sağlam.")
            print("    Model bu potansiyelin ne kadarını yakalıyor, aşağıda:")
            if 'model' in gg.index:
                md = gg.loc['model']['max_drawdown'] - bh['max_drawdown']
                cap = md / d_orc * 100 if d_orc != 0 else 0
                print(f"      model maxDD iyileştirmesi: {md*100:+.2f} puan "
                      f"→ oracle'ın %{cap:.0f}'i")
    if {'model', 'naive_vol'} <= set(gg.index):
        dm = gg.loc['model']['calmar'] - gg.loc['naive_vol']['calmar']
        print(f"\n  Model vs naif volatilite (Calmar farkı): {dm:+.3f}")
        if dm <= 0:
            print("  → Model, 'oynak hisselerden kaç' kuralını GEÇEMİYOR.")
            print("    51 özellikli derin modelin ekonomik katkısı yok; bunu")
            print("    jüri bulmadan siz raporlayın.")
        else:
            print("  → Model naif kuralı geçiyor: derin modelin ekonomik katkısı var.")

    print(f"\nÖzet   → {args.outdir}/portfolio_simulation.csv")
    print(f"Eğriler → {args.outdir}/portfolio_curves.csv")


if __name__ == '__main__':
    main()