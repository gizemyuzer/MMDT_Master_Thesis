"""
summarize_v2.py — düzeltilmiş koşuların tek sayfalık özeti.

results/ altındaki *_v2* dosyalarını ve purge denetimini okuyup
tezde kullanılacak biçimde basar. Ekran görüntüsü yerine bu çıktının
tamamını paylaşmak yeterlidir.

KULLANIM
    python summarize_v2.py
    python summarize_v2.py > ozet_v2.txt      # dosyaya yaz
"""
import os, glob
import numpy as np, pandas as pd

OUT = 'results'
pd.set_option('display.width', 200)


def sec(t):
    print("\n" + "═" * 78); print(t); print("═" * 78)


def show_portfolio(path):
    d = pd.read_csv(path)
    tag = os.path.basename(path).replace('portfolio_v2_', '').replace('.csv', '')
    sec(f"PORTFÖY — {tag}")
    if 'grp' not in d.columns:
        d['grp'] = d['strategy'].str.replace(r'_seed\d+', '', regex=True)
    for cb in sorted(d.cost_bps.unique()):
        g = (d[d.cost_bps == cb]
             .groupby('grp')[['cagr', 'max_drawdown', 'calmar', 'turnover']]
             .agg(['mean', 'std', 'count']))
        g.columns = ['_'.join(c) for c in g.columns]
        g = g.sort_values('calmar_mean', ascending=False)
        print(f"\n── {cb:.0f} bps ──")
        print(f"{'strateji':<26}{'CAGR':>9}{'maksDD':>10}{'Calmar':>9}"
              f"{'±sd':>8}{'devir':>8}{'n':>4}")
        print("-" * 74)
        for k in g.index:
            r = g.loc[k]
            sd = r.calmar_std if not np.isnan(r.calmar_std) else 0.0
            print(f"{k[:25]:<26}{100*r.cagr_mean:>8.2f}%{100*r.max_drawdown_mean:>9.2f}%"
                  f"{r.calmar_mean:>9.3f}{sd:>8.3f}{100*r.turnover_mean:>7.1f}%"
                  f"{int(r.calmar_count):>4}")


def show_cash(path):
    d = pd.read_csv(path)
    tag = os.path.basename(path).replace('cash_portfolio_v2_', '').replace('.csv', '')
    sec(f"NAKİT / TIMING VARYANTLARI — {tag}  (burn-in eşitlendi)")
    g = (d.groupby('mode')[['cagr', 'max_drawdown', 'calmar',
                            'avg_invested', 'avg_cash']].mean()
         .sort_values('calmar', ascending=False))
    print(f"{'mod':<18}{'CAGR':>9}{'maksDD':>10}{'Calmar':>9}"
          f"{'yatırım':>10}{'nakit':>9}")
    print("-" * 66)
    for k in g.index:
        r = g.loc[k]
        print(f"{k:<18}{100*r.cagr:>8.2f}%{100*r.max_drawdown:>9.2f}%"
              f"{r.calmar:>9.3f}{100*r.avg_invested:>9.1f}%{100*r.avg_cash:>8.1f}%")


def show_sweep(path):
    d = pd.read_csv(path)
    tag = os.path.basename(path).replace('pct_sweep_v2_', '').replace('.csv', '')
    sec(f"DIŞLAMA ORANI TARAMASI — {tag}")
    if 'grp' not in d.columns:
        d['grp'] = d['strategy'].str.replace(r'_seed\d+', '', regex=True)
    for m in ['calmar', 'max_drawdown', 'cagr']:
        print(f"\n{m}:")
        print(d.pivot_table(index='exclude_pct', columns='grp',
                            values=m).round(4).to_string())


def show_common(path):
    d = pd.read_csv(path)
    tag = os.path.basename(path).replace('common_sample_v2_', '').replace('.csv', '')
    sec(f"ORTAK ÖRNEKLEM — {tag}")
    if 'grp' not in d.columns:
        d['grp'] = d['model'].str.replace(r'_seed\d+', '', regex=True)
    g = d.groupby('grp')[['mcc', 'pr_auc', 'roc_auc', 'precision', 'recall', 'n']].mean()
    g['n_seed'] = d.groupby('grp').size()
    g = g.sort_values('mcc', ascending=False)
    print(f"{'model':<32}{'MCC':>9}{'PR-AUC':>9}{'ROC':>9}{'prec':>8}{'rec':>8}{'n_obs':>9}{'seed':>5}")
    print("-" * 80)
    for k in g.index:
        r = g.loc[k]
        print(f"{k[:31]:<32}{r.mcc:>9.4f}{r.pr_auc:>9.4f}{r.roc_auc:>9.4f}"
              f"{r.precision:>8.4f}{r.recall:>8.4f}{int(r.n):>9,}{int(r.n_seed):>5}")


def show_purge(path):
    d = pd.read_csv(path)
    sec("PURGE DENETİMİ")
    print(d.to_string(index=False))


def main():
    found = 0
    for f in sorted(glob.glob(os.path.join(OUT, 'portfolio_v2_*.csv'))):
        show_portfolio(f); found += 1
    for f in sorted(glob.glob(os.path.join(OUT, 'cash_portfolio_v2_*.csv'))):
        show_cash(f); found += 1
    for f in sorted(glob.glob(os.path.join(OUT, 'pct_sweep_v2_*.csv'))):
        show_sweep(f); found += 1
    for f in sorted(glob.glob(os.path.join(OUT, 'common_sample_v2_*.csv'))):
        show_common(f); found += 1
    p = os.path.join(OUT, 'purge_audit.csv')
    if os.path.exists(p):
        show_purge(p); found += 1

    # eski adlandırmayla kalmış dosyalar
    for old in ['portfolio_v2.csv', 'cash_portfolio_v2.csv',
                'pct_sweep_v2.csv', 'common_sample_v2.csv']:
        q = os.path.join(OUT, old)
        if os.path.exists(q):
            print(f"\n⚠️  {old} eski adlandırmayla duruyor (hangi hücreden "
                  f"geldiği belirsiz). run_portfolio_v2.py güncellendi; "
                  f"koşuları tekrarlayın.")

    if not found:
        print("Hiç _v2 sonucu bulunamadı. Önce koşuları çalıştırın.")
    else:
        sec("KAYNAK DOSYALAR")
        for f in sorted(glob.glob(os.path.join(OUT, '*_v2*.csv'))) + \
                 [os.path.join(OUT, 'purge_audit.csv')]:
            if os.path.exists(f):
                print(f"  {f}  ({os.path.getsize(f):,} bayt)")


if __name__ == '__main__':
    main()