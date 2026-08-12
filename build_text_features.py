"""
build_text_features.py
───────────────────────
EDGAR ham verisini DÖRDÜNCÜ MODALİTEYE (IV) çevirir.

═══════════════════════════════════════════════════════════════════════
İKİ KATMAN — bilinçli olarak ayrı
═══════════════════════════════════════════════════════════════════════

KATMAN A — 8-K OLAY BAYRAKLARI            (bedava, metin gerekmez)
    Kaynak: datasets/edgar_index.csv'deki `items` kolonu.
    SEC item kodlarını submissions JSON'ında hazır veriyor. Hiçbir dosya
    indirmeden, hiçbir NLP çalıştırmadan sıkıntı olayları elde edilir.
    Katman B çuvallarsa bile elinizde savunulabilir bir modalite kalır.

KATMAN B — METİN ÖZELLİKLERİ              (10-K/10-Q gerektirir)
    B1. Loughran-McDonald sözlük oranları (finansal metne özel duygu)
    B2. ARDIŞIK DOSYALAR ARASI BENZERLİK

    B2 literatür dayanağı: Cohen, Malloy & Nguyen (2020), "Lazy Prices",
    Journal of Finance 75(3). 10-K/10-Q dilindeki DEĞİŞİM gelecek getirileri
    tahmin ediyor; en çok değişen dosyalara sahip firmalar sonradan
    düşük performans gösteriyor. Yani seviye değil, FARK bilgi taşıyor —
    bu yüzden benzerlik skoru ayrı bir özellik olarak üretilir.

═══════════════════════════════════════════════════════════════════════
POINT-IN-TIME DOĞRULUĞU
═══════════════════════════════════════════════════════════════════════
Her özellik `filing_date` ile hizalanır, `period_of_report` ile DEĞİL.
Bir 10-K mali yıl bitiminden 60-90 gün sonra yayımlanır; period_of_report
kullanmak henüz var olmayan bir belgeyi modele vermek olurdu — rdq/datadate
ayrımında düzelttiğiniz hatanın aynısı.

Yuvarlanan pencereler yalnızca GEÇMİŞE bakar (rolling, shift yok gerekmez
çünkü filing_date zaten olayın kamuya açıldığı gün).

═══════════════════════════════════════════════════════════════════════
LOUGHRAN-McDONALD SÖZLÜĞÜ
═══════════════════════════════════════════════════════════════════════
Genel amaçlı duygu sözlükleri finansal metinde yanılır: "liability",
"tax", "cost", "capital" günlük dilde olumsuzdur ama bilançoda nötrdür.
LM sözlüğü 10-K'lar üzerinde kurulmuştur, bu yüzden standarttır.

İNDİRME (bir kez, elle):
    https://sraf.nd.edu/loughranmcdonald-master-dictionary/
    "Loughran-McDonald Master Dictionary" CSV dosyasını indirip şuraya koyun:
        datasets/LM_MasterDictionary.csv

KULLANIM:
    python build_text_features.py --stage events     # Katman A — hemen çalışır
    python build_text_features.py --stage text       # Katman B — indirme bitince
    python build_text_features.py --self-test        # sentetik doğrulama

Çıktı: datasets/text_features.csv   (date, ticker, <özellikler>)
"""
import os
import re
import gzip
import argparse
from collections import Counter

import numpy as np
import pandas as pd

INDEX = os.path.join('datasets', 'edgar_index.csv')
MANIFEST = os.path.join('datasets', 'edgar_manifest.csv')
LM_DICT = os.path.join('datasets', 'LM_MasterDictionary.csv')
OUT = os.path.join('datasets', 'text_features.csv')

# ── Katman A: 8-K sıkıntı item'ları ──
# Her biri ayrı bir özellik; toplulaştırmak bilgi kaybettirir çünkü
# 4.02 (yeniden düzenleme) ile 5.02 (yönetici ayrılığı) çok farklı olaylar.
EVENT_SPECS = {
    'EK_Bankruptcy':    ('1.03', 252),   # iflas — 1 yıl hafıza
    'EK_DebtTrigger':   ('2.04', 252),   # covenant ihlali
    'EK_Impairment':    ('2.06', 252),   # değer düşüklüğü
    'EK_ListingWarn':   ('3.01', 252),   # delisting uyarısı
    'EK_AuditorChange': ('4.01', 252),   # denetçi değişikliği
    'EK_Restatement':   ('4.02', 252),   # mali tablolara güvenilemez
    'EK_MgmtChange':    ('5.02', 126),   # yönetim değişikliği — 6 ay
}
ACTIVITY_WINDOW = 60      # genel 8-K yoğunluğu penceresi

_WORD = re.compile(r"[A-Za-z']+")


# ══════════════════════════════════════════════════════════════════
# KATMAN A — 8-K olay bayrakları
# ══════════════════════════════════════════════════════════════════
def build_event_features(idx: pd.DataFrame, panel_dates: pd.DatetimeIndex,
                         tickers) -> pd.DataFrame:
    """
    Her (tarih, hisse) için, GEÇMİŞ penceredeki 8-K olay sayıları.

    Yuvarlanan toplam kullanılır, "en son ne zaman oldu" değil: bir firma
    aynı yıl iki kez yeniden düzenleme ilan ettiyse bu tek seferden daha
    kötü bir sinyaldir ve sayım bunu korur.
    """
    ek = idx[idx['form'].astype(str).str.startswith('8-K')].copy()
    if ek.empty:
        print("  ⚠️ Hiç 8-K yok — index'i --forms 10-K 10-Q 8-K ile yeniden üretin")
        return pd.DataFrame()

    ek['filing_date'] = pd.to_datetime(ek['filing_date'])
    ek['items'] = ek['items'].fillna('').astype(str)

    frames = []
    for tic, g in ek.groupby('ticker'):
        if tic not in tickers:
            continue
        # Günlük olay göstergeleri → işlem günlerine yeniden indeksle
        daily = pd.DataFrame(index=panel_dates)
        counts = g.groupby('filing_date').size()
        daily['_any'] = counts.reindex(panel_dates, fill_value=0).fillna(0)

        for name, (code, _win) in EVENT_SPECS.items():
            hit = g[g['items'].str.contains(code, regex=False)]
            c = hit.groupby('filing_date').size() if len(hit) else pd.Series(dtype=float)
            daily[name] = (c.reindex(panel_dates, fill_value=0).fillna(0)
                           if len(c) else 0.0)

        out = pd.DataFrame(index=panel_dates)
        for name, (_code, win) in EVENT_SPECS.items():
            out[name] = daily[name].rolling(win, min_periods=1).sum()
        out['EK_Activity_60d'] = daily['_any'].rolling(ACTIVITY_WINDOW, min_periods=1).sum()

        # Son 8-K'dan bu yana geçen gün — yakınlık ayrı bir bilgi
        last = daily['_any'].where(daily['_any'] > 0)
        idx_num = pd.Series(np.arange(len(panel_dates)), index=panel_dates)
        last_seen = idx_num.where(last.notna()).ffill()
        out['EK_DaysSince'] = (idx_num - last_seen).fillna(999).clip(upper=999)

        out['Ticker'] = tic
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    res = pd.concat(frames)
    res.index.name = 'date'
    return res.reset_index()


# ══════════════════════════════════════════════════════════════════
# KATMAN B — Loughran-McDonald + benzerlik
# ══════════════════════════════════════════════════════════════════
def load_lm():
    if not os.path.exists(LM_DICT):
        raise SystemExit(
            f"\n{LM_DICT} bulunamadı.\n\n"
            "Loughran-McDonald Master Dictionary'yi indirin:\n"
            "  https://sraf.nd.edu/loughranmcdonald-master-dictionary/\n"
            f"ve CSV'yi şu adla kaydedin: {LM_DICT}\n\n"
            "Neden genel bir duygu sözlüğü değil: 'liability', 'cost', 'tax'\n"
            "günlük dilde olumsuzdur ama finansal metinde nötrdür. LM sözlüğü\n"
            "10-K'lar üzerinde kurulduğu için bu hatayı yapmaz.\n"
        )
    d = pd.read_csv(LM_DICT)
    d.columns = [c.strip().lower() for c in d.columns]
    word_col = 'word' if 'word' in d.columns else d.columns[0]
    d[word_col] = d[word_col].astype(str).str.lower()
    cats = {}
    for cat in ('negative', 'positive', 'uncertainty', 'litigious', 'constraining'):
        if cat in d.columns:
            cats[cat] = set(d.loc[pd.to_numeric(d[cat], errors='coerce').fillna(0) > 0,
                                  word_col])
    if not cats:
        raise SystemExit(f"{LM_DICT} beklenen kategori kolonlarını içermiyor: {list(d.columns)[:12]}")
    print(f"  LM sözlüğü: " + " | ".join(f"{k}={len(v):,}" for k, v in cats.items()))
    return cats


def lm_scores(text: str, cats: dict) -> dict:
    words = _WORD.findall(text.lower())
    n = len(words)
    out = {'LM_WordCount': float(n)}
    if n == 0:
        for c in cats:
            out[f'LM_{c.capitalize()}'] = np.nan
        return out
    cnt = Counter(words)
    for c, vocab in cats.items():
        out[f'LM_{c.capitalize()}'] = sum(v for w, v in cnt.items() if w in vocab) / n
    return out


def cosine_similarity_counts(a: Counter, b: Counter) -> float:
    """İki kelime sayımı arasında kosinüs benzerliği (Cohen ve ark. 2020)."""
    if not a or not b:
        return np.nan
    common = set(a) & set(b)
    dot = sum(a[w] * b[w] for w in common)
    na = np.sqrt(sum(v * v for v in a.values()))
    nb = np.sqrt(sum(v * v for v in b.values()))
    return float(dot / (na * nb)) if na and nb else np.nan


def build_text_features(man: pd.DataFrame, cats: dict) -> pd.DataFrame:
    """Her 10-K/10-Q için LM oranları + bir önceki AYNI TÜR dosyayla benzerlik."""
    man = man[(man['status'] == 'ok') &
              (man['form'].astype(str).str.startswith('10-'))].copy()
    man['filing_date'] = pd.to_datetime(man['filing_date'])
    man = man.sort_values(['ticker', 'form', 'filing_date'])

    rows = []
    prev_counts = {}          # (ticker, form) → önceki dosyanın Counter'ı
    total = len(man)
    for i, r in enumerate(man.itertuples(), 1):
        if not r.path or not os.path.exists(r.path):
            continue
        try:
            with gzip.open(r.path, 'rt', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
        except OSError:
            continue

        rec = {'ticker': r.ticker, 'form': r.form, 'filing_date': r.filing_date}
        rec.update(lm_scores(txt, cats))

        # Benzerlik YALNIZCA aynı form tipiyle karşılaştırılır: bir 10-K ile
        # 10-Q'yu kıyaslamak yapısal fark yüzünden yapay düşük benzerlik verir.
        words = _WORD.findall(txt.lower())
        c = Counter(words)
        key = (r.ticker, r.form)
        rec['TXT_SimPrev'] = cosine_similarity_counts(prev_counts.get(key), c)
        prev = prev_counts.get(key)
        rec['TXT_LenChange'] = (np.log(len(words) / sum(prev.values()))
                                if prev and sum(prev.values()) > 0 and len(words) > 0
                                else np.nan)
        prev_counts[key] = c
        rows.append(rec)

        if i % 500 == 0 or i == total:
            print(f"    [{i:,}/{total:,}] işlendi")

    return pd.DataFrame(rows)


def align_to_panel(feat: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """
    Dosya-seviyesi özellikleri panele POINT-IN-TIME yayar.
    merge_asof(direction='backward') → her gün, o güne kadar YAYIMLANMIŞ
    en son dosyanın değerini alır. İleriye bakış yok.
    """
    # .values ile indeks hizalamasını KIRARAK kur. panel'in indeksi zaten
    # 'date' adını taşıyor; üstüne 'date' kolonu eklemek pandas'ta
    # "both an index level and a column label" hatasına yol açıyor.
    p = pd.DataFrame({'ticker': panel['Ticker'].values,
                      'date': pd.to_datetime(panel.index.values)})
    p = p.sort_values('date').reset_index(drop=True)
    f = feat.sort_values('filing_date')
    cols = [c for c in f.columns if c not in ('ticker', 'form', 'filing_date')]

    out = pd.merge_asof(p, f[['ticker', 'filing_date'] + cols],
                        left_on='date', right_on='filing_date', by='ticker',
                        direction='backward')
    out['TXT_DaysSinceFiling'] = (out['date'] - out['filing_date']).dt.days
    return out.drop(columns=['filing_date'])


# ══════════════════════════════════════════════════════════════════
def self_test():
    print("═" * 74)
    print("SENTETİK DOĞRULAMA")
    print("═" * 74)
    ok = True

    # 1) LM oranları
    cats = {'negative': {'loss', 'decline', 'litigation'},
            'positive': {'gain', 'strong'}}
    s = lm_scores("Loss and decline and gain . The litigation continues", cats)
    exp_neg, exp_pos = 3 / 8, 1 / 8
    t1 = abs(s['LM_Negative'] - exp_neg) < 1e-9 and abs(s['LM_Positive'] - exp_pos) < 1e-9
    print(f"  [{'GEÇTİ' if t1 else 'KALDI'}] LM oranı  neg={s['LM_Negative']:.4f} "
          f"(beklenen {exp_neg:.4f}) | kelime={s['LM_WordCount']:.0f}")
    ok &= t1

    # 2) Benzerlik: aynı metin → 1.0, ortak kelime yok → 0.0
    a = Counter("risk risk debt".split())
    b = Counter("risk risk debt".split())
    c = Counter("growth profit".split())
    t2 = abs(cosine_similarity_counts(a, b) - 1.0) < 1e-9 and \
         abs(cosine_similarity_counts(a, c) - 0.0) < 1e-9
    print(f"  [{'GEÇTİ' if t2 else 'KALDI'}] benzerlik aynı={cosine_similarity_counts(a,b):.4f} "
          f"farklı={cosine_similarity_counts(a,c):.4f}")
    ok &= t2

    # 3) Olay penceresi: 2 gün önceki 4.02, 252 gün penceresinde görünmeli;
    #    400 gün öncekiler görünmemeli
    dates = pd.bdate_range('2020-01-01', '2021-12-31')
    idx = pd.DataFrame([
        {'ticker': 'AAA', 'form': '8-K', 'items': '4.02', 'filing_date': '2021-06-01'},
        {'ticker': 'AAA', 'form': '8-K', 'items': '2.02', 'filing_date': '2021-06-02'},
        {'ticker': 'AAA', 'form': '8-K', 'items': '4.02', 'filing_date': '2020-01-02'},
    ])
    ev = build_event_features(idx, dates, {'AAA'})
    ev = ev.set_index('date')
    after = ev.loc['2021-06-03', 'EK_Restatement']
    long_after = ev.loc['2021-12-31', 'EK_Restatement']   # 2020-01 olayı düşmüş olmalı
    before = ev.loc['2021-05-28', 'EK_Restatement']
    t3 = after == 1.0 and before == 0.0 and long_after == 1.0
    print(f"  [{'GEÇTİ' if t3 else 'KALDI'}] olay penceresi  olaydan önce={before:.0f} "
          f"sonra={after:.0f} 7ay sonra={long_after:.0f}")
    ok &= t3

    # 4) İLERİYE BAKIŞ YOK — en kritik test
    t4 = before == 0.0
    print(f"  [{'GEÇTİ' if t4 else 'KALDI'}] ileriye bakış yok (olaydan önceki gün sıfır)")
    ok &= t4

    print("\n" + ("  ✓ Tüm testler geçti." if ok else "  ✗ BAŞARISIZ — çıktıyı kullanmayın."))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['events', 'text', 'all'], default='all')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)

    from datasets.feature_engineering import prepare_dataset
    panel = prepare_dataset(force_refresh=False)
    dates = pd.DatetimeIndex(sorted(panel.index.unique()))
    tickers = set(panel['Ticker'].unique())
    print(f"Panel: {len(panel):,} satır | {len(tickers)} hisse | "
          f"{dates.min().date()} → {dates.max().date()}")

    parts = []

    if args.stage in ('events', 'all'):
        print("\n[A] 8-K olay bayrakları")
        if not os.path.exists(INDEX):
            raise SystemExit(f"{INDEX} yok — önce fetch_edgar_filings.py --stage index")
        idx = pd.read_csv(INDEX)
        if 'items' not in idx.columns:
            raise SystemExit(
                f"{INDEX} 'items' kolonu içermiyor.\n"
                "8-K desteği eklenmeden üretilmiş. Yeniden koşun:\n"
                "  python fetch_edgar_filings.py --stage index "
                "--forms 10-K 10-Q 8-K --overwrite")
        ev = build_event_features(idx, dates, tickers)
        if not ev.empty:
            parts.append(ev)
            print(f"  {len(ev):,} satır | kolonlar: "
                  f"{[c for c in ev.columns if c not in ('date','Ticker')]}")
            nz = {c: int((ev[c] > 0).sum()) for c in EVENT_SPECS}
            print("  sıfırdan farklı gözlem sayısı:")
            for k, v in sorted(nz.items(), key=lambda x: -x[1]):
                print(f"    {k:<20}{v:>10,}")

    if args.stage in ('text', 'all'):
        print("\n[B] 10-K/10-Q metin özellikleri")
        if not os.path.exists(MANIFEST):
            print(f"  {MANIFEST} yok — indirme bitmemiş, bu katman atlanıyor")
        else:
            cats = load_lm()
            man = pd.read_csv(MANIFEST)
            tf = build_text_features(man, cats)
            if not tf.empty:
                al = align_to_panel(tf, panel)
                parts.append(al.rename(columns={'ticker': 'Ticker'}))
                print(f"  {len(tf):,} dosya işlendi | {tf.ticker.nunique()} firma")
                print(f"  ortalama LM_Negative = {tf['LM_Negative'].mean():.4f}")
                if 'TXT_SimPrev' in tf:
                    print(f"  ortalama benzerlik   = {tf['TXT_SimPrev'].mean():.4f}")

    if not parts:
        raise SystemExit("Hiç özellik üretilmedi.")

    res = parts[0]
    for p in parts[1:]:
        res = res.merge(p, on=['date', 'Ticker'], how='outer')
    res.to_csv(OUT, index=False)
    print(f"\n{len(res):,} satır × {res.shape[1]} kolon → {OUT}")
    print("\nSONRAKİ ADIM: bu kolonları feature_engineering.py'ye TEXT_COLS")
    print("grubu olarak ekleyip Δ(IV | I+II) faktöriyel hücresini koşmak.")


if __name__ == '__main__':
    main()