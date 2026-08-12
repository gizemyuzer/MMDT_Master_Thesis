"""
fetch_edgar_filings.py
───────────────────────
SEC EDGAR 10-K / 10-Q METİN VERİSİ — dördüncü modalite (IV) için ham veri.

═══════════════════════════════════════════════════════════════════════
NEDEN
═══════════════════════════════════════════════════════════════════════
Şu ana kadarki üç modalite de SAYISAL: fiyat, muhasebe oranı, makro seri.
ML literatüründe "multi-modal" genelde farklı veri TİPLERİ demektir. Metin
eklemek tezin çerçevesini gerçekten çok-modaliteli hale getiriyor.

═══════════════════════════════════════════════════════════════════════
KRİTİK TASARIM KARARLARI
═══════════════════════════════════════════════════════════════════════

1) CIK EŞLEMESİ permno → gvkey → cik ZİNCİRİYLE YAPILIR
   SEC'in güncel company_tickers.json dosyası SADECE AKTİF şirketleri
   içerir. Evreninizin 395 firmasından 164'ü delist oldu — onları ticker
   üzerinden bulmak imkânsız. Compustat'ın company tablosunda cik alanı
   var ve delist olmuş firmalar orada duruyor. Bu yüzden zincir:
       universe.csv (permno) → crsp.ccmxpf_lnkhist (gvkey) → comp.company (cik)
   Ticker üzerinden eşleme yapmak hayatta kalma yanlılığını METİN
   modalitesine geri sokardı — tüm veri tasarımınızı boşa çıkarırdı.

2) HİZALAMA filingDate İLE, periodOfReport İLE DEĞİL
   Bir 10-K, mali yıl bitiminden 60-90 gün sonra yayımlanır. periodOfReport
   kullanmak, henüz yayımlanmamış bir belgeyi modele vermek olurdu — tam
   olarak rdq/datadate ayrımında düzelttiğiniz look-ahead hatası.
   Manifest'te ikisi de saklanır, hizalamada filingDate kullanılır.

3) METİN HAM SAKLANIR (gzip)
   Loughran-McDonald sözlük sayımı ucuz, FinBERT gömme vektörleri pahalı.
   İkisini de sonradan yapabilmek için temizlenmiş metin diske yazılır.
   Yalnızca sayımları saklamak, FinBERT aşamasında her şeyi yeniden
   indirmeyi gerektirirdi.

4) HIZ SINIRI
   SEC saniyede 10 istekten fazlasını engeller ve iletişim bilgisi içeren
   bir User-Agent zorunlu tutar. Varsayılan 5 istek/sn — engellenmemek
   için bilinçli olarak muhafazakâr.

═══════════════════════════════════════════════════════════════════════
BEKLENEN HACİM
═══════════════════════════════════════════════════════════════════════
~395 firma × 15 yıl ≈ 12.000-16.000 dosya (10-K + 10-Q).
Sıkıştırılmış disk kullanımı ≈ 2-4 GB. Süre ≈ 4-8 saat.
Kesilirse kaldığı yerden devam eder (--resume varsayılan açık).

KULLANIM:
    # ZORUNLU: SEC iletişim bilgisi istiyor
    export SEC_USER_AGENT="Gizem Yuzer gizemyuzer1@gmail.com"

    python fetch_edgar_filings.py --stage cik      # 1) CIK eşlemesi (WRDS)
    python fetch_edgar_filings.py --stage index    # 2) dosya listesi (SEC)
    python fetch_edgar_filings.py --stage download # 3) indirme + temizleme
    python fetch_edgar_filings.py                  # üçü sırayla

    python fetch_edgar_filings.py --forms 10-K     # sadece yıllık
    python fetch_edgar_filings.py --limit 5        # deneme: 5 firma

Çıktılar:
    datasets/edgar_cik_map.csv     permno · gvkey · cik · ticker
    datasets/edgar_index.csv       her dosya: accession, form, tarihler, url
    datasets/edgar_manifest.csv    indirilenler: yol, karakter sayısı, durum
    data/edgar/<cik>/<accession>.txt.gz
"""
import os
import re
import gzip
import json
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import requests

UNIVERSE = os.path.join('datasets', 'universe.csv')
CIK_MAP = os.path.join('datasets', 'edgar_cik_map.csv')
INDEX = os.path.join('datasets', 'edgar_index.csv')
MANIFEST = os.path.join('datasets', 'edgar_manifest.csv')
TEXT_DIR = os.path.join('data', 'edgar')

START_DATE, END_DATE = '2010-01-01', '2024-12-31'

# ══════════════════════════════════════════════════════════════════
# 8-K item kodları — sıkıntı (distress) sinyalleri
# ══════════════════════════════════════════════════════════════════
# 8-K, önemli bir olaydan sonraki 4 iş günü içinde ZORUNLU olarak
# dosyalanır. Yani birincil kaynaktan, tam kapsamlı, delist olan firmalar
# dahil bir "haber" akışı. Ticari haber veritabanları küçük ölçekli
# hisselerde zayıf kapsama sunar ve 2010'a kadar geriye gitmez; 8-K her
# ikisini de çözer.
#
# Item 4.02 muhasebe literatüründe en güvenilir sıkıntı öngörücülerinden
# biridir (mali tabloların yeniden düzenlenmesi ilanı). Item 3.01 doğrudan
# delisting'in habercisidir — ki evreninizin %41'i delist oldu.
DISTRESS_ITEMS = {
    '1.03': 'İflas / konkordato',
    '2.04': 'Borcu hızlandıran tetikleyici olay (covenant ihlali)',
    '2.06': 'Maddi değer düşüklüğü (impairment)',
    '3.01': 'Kotasyon kuralı ihlali / delisting uyarısı',
    '4.01': 'Bağımsız denetçi değişikliği',
    '4.02': 'Önceki mali tablolara güvenilemez (yeniden düzenleme)',
    '5.02': 'Üst yönetim ayrılığı / atama',
}
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SHARD = "https://data.sec.gov/submissions/{name}"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


_UA_OVERRIDE = None


def user_agent():
    """Önce --user-agent bayrağı, sonra ortam değişkeni."""
    ua = (_UA_OVERRIDE or os.environ.get('SEC_USER_AGENT', '')).strip()
    if not ua or '@' not in ua:
        raise SystemExit(
            "\nSEC iletişim bilgisi gerekli (e-posta içermeli).\n"
            "SEC, kimliksiz istekleri engeller.\n\n"
            "  En kolayı — komut satırında verin:\n"
            '    python fetch_edgar_filings.py --user-agent "Gizem Yuzer gizemyuzer1@gmail.com"\n\n'
            "  Ya da ortam değişkeni olarak:\n"
            '    PowerShell : $env:SEC_USER_AGENT = "Gizem Yuzer gizemyuzer1@gmail.com"\n'
            '    CMD        : set SEC_USER_AGENT=Gizem Yuzer gizemyuzer1@gmail.com\n'
            '    bash/zsh   : export SEC_USER_AGENT="Gizem Yuzer gizemyuzer1@gmail.com"\n'
        )
    return ua


class Throttle:
    """SEC saniyede 10 istek sınırı — varsayılan 5/sn ile güvenli tarafta."""

    def __init__(self, per_second=5.0):
        self.min_gap = 1.0 / per_second
        self.last = 0.0

    def wait(self):
        gap = time.time() - self.last
        if gap < self.min_gap:
            time.sleep(self.min_gap - gap)
        self.last = time.time()


def get(url, throttle, session, tries=4, timeout=30):
    """Üstel geri çekilmeli GET. Başarısızlıkta None döner, PATLAMAZ —
    12 bin dosyalık bir koşu tek bir 503 yüzünden durmamalı."""
    for attempt in range(tries):
        throttle.wait()
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return None


# ══════════════════════════════════════════════════════════════════
# AŞAMA 1 — CIK eşlemesi
# ══════════════════════════════════════════════════════════════════
def stage_cik():
    print("═" * 78)
    print("AŞAMA 1 — CIK eşlemesi (permno → gvkey → cik)")
    print("═" * 78)

    uni = pd.read_csv(UNIVERSE)
    permnos = sorted(uni['permno'].dropna().astype(int).unique().tolist())
    print(f"  Evren: {len(permnos)} permno")

    import wrds
    db = wrds.Connection(wrds_username='gizemyuzer')
    try:
        # CRSP ↔ Compustat bağlantı tablosu.
        # linktype LU/LC ve linkprim P/C → standart araştırma filtresi
        # (Compustat'ın "birincil" bağlantısı; diğerleri ikincil sınıflar).
        lnk = db.raw_sql(f"""
            SELECT lpermno AS permno, gvkey, linkdt, linkenddt
            FROM crsp.ccmxpf_lnkhist
            WHERE linktype IN ('LU','LC') AND linkprim IN ('P','C')
              AND lpermno IN ({','.join(map(str, permnos))})
        """)
        comp = db.raw_sql("""
            SELECT gvkey, cik, conm
            FROM comp.company
            WHERE cik IS NOT NULL AND cik <> ''
        """)
    finally:
        db.close()

    print(f"  Bağlantı satırı: {len(lnk)} | cik'li Compustat firması: {len(comp)}")

    # Bağlantı dönemi test sonumuzla kesişmeli
    lnk['linkenddt'] = pd.to_datetime(lnk['linkenddt']).fillna(pd.Timestamp('2100-01-01'))
    lnk['linkdt'] = pd.to_datetime(lnk['linkdt'])
    lnk = lnk[(lnk['linkdt'] <= pd.Timestamp(END_DATE)) &
              (lnk['linkenddt'] >= pd.Timestamp(START_DATE))]

    m = lnk.merge(comp, on='gvkey', how='inner')
    m['cik'] = pd.to_numeric(m['cik'], errors='coerce')
    m = m.dropna(subset=['cik'])
    m['cik'] = m['cik'].astype(int)
    # Bir permno birden çok gvkey'e bağlanabilir (birleşmeler) — en uzun
    # süreli bağlantıyı tut, keyfi seçim yapma.
    m['span'] = (m['linkenddt'] - m['linkdt']).dt.days
    m = m.sort_values('span', ascending=False).drop_duplicates('permno', keep='first')

    out = uni[['ticker', 'permno', 'comnam']].merge(
        m[['permno', 'gvkey', 'cik', 'conm']], on='permno', how='left')
    out.to_csv(CIK_MAP, index=False)

    hit = out['cik'].notna().sum()
    print(f"\n  Eşleşen : {hit}/{len(out)}  (%{100*hit/len(out):.1f})")
    miss = out[out['cik'].isna()]
    if len(miss):
        print(f"  Eşleşmeyen {len(miss)} firma (ilk 15):")
        print("   ", ', '.join(miss['ticker'].head(15).tolist()))
        print("  Bunlar büyük ihtimalle Compustat kapsamı dışında veya çok erken")
        print("  delist olmuş firmalar. Metin modalitesinde eksik kalacaklar —")
        print("  tez metninde kapsama oranı olarak raporlanmalı.")
    print(f"\n  → {CIK_MAP}")


# ══════════════════════════════════════════════════════════════════
# AŞAMA 2 — dosya listesi
# ══════════════════════════════════════════════════════════════════
def stage_index(forms, rate, limit):
    print("═" * 78)
    print("AŞAMA 2 — SEC dosya listesi")
    print("═" * 78)

    if not os.path.exists(CIK_MAP):
        raise SystemExit(
            f"\n{CIK_MAP} yok. Önce CIK eşlemesini üretin:\n"
            f"    python fetch_edgar_filings.py --stage cik\n"
        )
    cm = pd.read_csv(CIK_MAP).dropna(subset=['cik'])
    cm['cik'] = cm['cik'].astype(int)
    if limit:
        cm = cm.head(limit)
    print(f"  {len(cm)} firma | formlar: {forms}")

    session = requests.Session()
    session.headers.update({'User-Agent': user_agent(),
                            'Accept-Encoding': 'gzip, deflate'})
    th = Throttle(rate)

    rows, failed = [], []
    for i, r in enumerate(cm.itertuples(), 1):
        cik = int(r.cik)
        resp = get(SUBMISSIONS.format(cik=cik), th, session)
        if resp is None:
            failed.append(r.ticker)
            continue
        try:
            js = resp.json()
        except json.JSONDecodeError:
            failed.append(r.ticker)
            continue

        blocks = [js.get('filings', {}).get('recent', {})]
        # Eski dosyalar ayrı parçalarda; 15 yıllık aralık için bunlar ŞART.
        for f in js.get('filings', {}).get('files', []):
            sh = get(SHARD.format(name=f['name']), th, session)
            if sh is not None:
                try:
                    blocks.append(sh.json())
                except json.JSONDecodeError:
                    pass

        n = 0
        for b in blocks:
            if not b or 'accessionNumber' not in b:
                continue
            for j in range(len(b['accessionNumber'])):
                form = b['form'][j]
                if form not in forms:
                    continue
                fdate = b['filingDate'][j]
                if not (START_DATE <= fdate <= END_DATE):
                    continue
                # 8-K'larda SEC, item kodlarını submissions JSON'ında HAZIR verir
                # ("2.02,9.01" gibi). Yani sıkıntı olaylarını metni hiç
                # indirmeden, ayrıştırmadan, ikili özellik olarak elde ediyoruz.
                items = b.get('items', [''] * (j + 1))[j] if 'items' in b else ''
                rows.append({
                    'ticker': r.ticker, 'permno': r.permno, 'cik': cik,
                    'form': form,
                    'accession': b['accessionNumber'][j],
                    'filing_date': fdate,                    # HİZALAMA BUNUNLA
                    'period_of_report': b.get('reportDate', [''] * (j + 1))[j],
                    'primary_doc': b.get('primaryDocument', [''] * (j + 1))[j],
                    'items': items,
                })
                n += 1
        if i % 25 == 0 or i == len(cm):
            print(f"  [{i}/{len(cm)}] {r.ticker:<6} +{n:<4} toplam {len(rows):,}")

    idx = pd.DataFrame(rows)
    if idx.empty:
        print("  ⚠️ Hiç dosya bulunamadı.")
        return
    idx['acc_nodash'] = idx['accession'].str.replace('-', '', regex=False)
    idx['url'] = [ARCHIVE.format(cik=c, acc=a, doc=d)
                  for c, a, d in zip(idx.cik, idx.acc_nodash, idx.primary_doc)]
    idx = idx[idx['primary_doc'].astype(str).str.len() > 0]
    idx.to_csv(INDEX, index=False)

    print(f"\n  {len(idx):,} dosya | {idx.ticker.nunique()} firma")
    print(idx.groupby('form').size().to_string())
    print(f"  Yıl aralığı: {idx.filing_date.min()} → {idx.filing_date.max()}")

    # ── 8-K sıkıntı item'ları — metin işlemeden hazır özellik ──
    ek = idx[idx['form'].astype(str).str.startswith('8-K')]
    if len(ek):
        print(f"\n  8-K SIKINTI OLAYLARI ({len(ek):,} dosya içinde):")
        print(f"  {'item':<8}{'adet':>8}  açıklama")
        print("  " + "-" * 62)
        for code, desc in sorted(DISTRESS_ITEMS.items()):
            n = ek['items'].astype(str).str.contains(code, regex=False).sum()
            if n:
                print(f"  {code:<8}{n:>8}  {desc}")
        print("\n  Bunlar ikili özellik olarak doğrudan kullanılabilir —")
        print("  hiçbir metin indirmeye veya NLP'ye gerek yok.")
    if failed:
        print(f"  ⚠️ {len(failed)} firma için liste alınamadı: {failed[:10]}")
    print(f"\n  → {INDEX}")


# ══════════════════════════════════════════════════════════════════
# AŞAMA 3 — indirme + temizleme
# ══════════════════════════════════════════════════════════════════
_TAG = re.compile(r'<[^>]+>')
_SCRIPT = re.compile(r'<(script|style)[^>]*>.*?</\1>', re.I | re.S)
_WS = re.compile(r'[ \t\r\f\v]+')
_NL = re.compile(r'\n{3,}')


def clean_html(raw: str) -> str:
    """HTML → düz metin. bs4 varsa onu kullanır, yoksa regex'e düşer."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, 'lxml')
        for t in soup(['script', 'style']):
            t.decompose()
        txt = soup.get_text('\n')
    except Exception:
        txt = _TAG.sub(' ', _SCRIPT.sub(' ', raw))
    txt = (txt.replace('&nbsp;', ' ').replace('&amp;', '&')
              .replace('&lt;', '<').replace('&gt;', '>').replace('&#160;', ' '))
    # bs4 &nbsp;'i \xa0'ya çevirir, yukarıdaki replace'ten ÖNCE. Bırakılırsa
    # kelime ayırıcı sayılmaz ve "Item\xa01A" gibi başlıklar regex'le
    # bulunamaz — item ayrıştırma sessizce başarısız olur.
    txt = txt.replace('\xa0', ' ').replace('’', "'").replace('“', '"') \
             .replace('”', '"').replace('—', '-').replace('–', '-')
    return _NL.sub('\n\n', _WS.sub(' ', txt)).strip()


def stage_download(rate, limit, overwrite, download_forms=None):
    print("═" * 78)
    print("AŞAMA 3 — indirme + temizleme")
    print("═" * 78)

    if not os.path.exists(INDEX):
        raise SystemExit(
            f"\n{INDEX} yok. Önce dosya listesini üretin:\n"
            f"    python fetch_edgar_filings.py --stage index\n"
        )
    idx = pd.read_csv(INDEX)

    # 8-K'ların METNİ gerekmiyor: item kodları zaten index'te hazır geliyor.
    # Gövdeleri çoğunlukla standart ifade ve ek dosya; binlercesini indirmek
    # saatler alır ve hiçbir yeni bilgi vermez. Varsayılan olarak sadece
    # 10-K/10-Q indirilir.
    if download_forms:
        before = len(idx)
        idx = idx[idx['form'].astype(str).isin(download_forms)]
        skipped = before - len(idx)
        print(f"  Form filtresi: {download_forms} → {len(idx):,} dosya "
              f"({skipped:,} atlandı, ör. 8-K metni gerekmiyor)")

    if limit:
        keep = idx['ticker'].drop_duplicates().head(limit)
        idx = idx[idx['ticker'].isin(keep)]

    done = {}
    if os.path.exists(MANIFEST) and not overwrite:
        prev = pd.read_csv(MANIFEST)
        done = {a: True for a in prev['accession']}
        print(f"  [Resume] {len(done):,} dosya zaten indirilmiş")
        rows = prev.to_dict('records')
    else:
        rows = []

    todo = idx[~idx['accession'].isin(done)]
    print(f"  İndirilecek: {len(todo):,} / {len(idx):,}")
    if todo.empty:
        print("  Yapacak bir şey yok.")
        return

    session = requests.Session()
    session.headers.update({'User-Agent': user_agent(),
                            'Accept-Encoding': 'gzip, deflate'})
    th = Throttle(rate)
    t0 = time.time()
    ok = fail = 0

    for i, r in enumerate(todo.itertuples(), 1):
        d = os.path.join(TEXT_DIR, str(int(r.cik)))
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{r.accession}.txt.gz")

        resp = get(r.url, th, session, timeout=60)
        if resp is None:
            rows.append({'accession': r.accession, 'ticker': r.ticker,
                         'cik': r.cik, 'form': r.form,
                         'filing_date': r.filing_date,
                         'period_of_report': r.period_of_report,
                         'path': '', 'n_chars': 0, 'status': 'download_failed'})
            fail += 1
        else:
            txt = clean_html(resp.text)
            with gzip.open(path, 'wt', encoding='utf-8') as f:
                f.write(txt)
            rows.append({'accession': r.accession, 'ticker': r.ticker,
                         'cik': r.cik, 'form': r.form,
                         'filing_date': r.filing_date,
                         'period_of_report': r.period_of_report,
                         'path': path, 'n_chars': len(txt), 'status': 'ok'})
            ok += 1

        if i % 200 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(MANIFEST, index=False)
            el = time.time() - t0
            eta = (el / i) * (len(todo) - i) / 60
            print(f"  [{i:,}/{len(todo):,}] ok={ok:,} hata={fail:,} "
                  f"| {el/60:.0f} dk geçti | ~{eta:.0f} dk kaldı")

    pd.DataFrame(rows).to_csv(MANIFEST, index=False)
    M = pd.DataFrame(rows)
    good = M[M.status == 'ok']
    print(f"\n  Başarılı: {len(good):,} | Hatalı: {(M.status != 'ok').sum():,}")
    if len(good):
        print(f"  Ortalama uzunluk: {good.n_chars.mean():,.0f} karakter")
        print(f"  Medyan  uzunluk: {good.n_chars.median():,.0f} karakter")
        print(f"  Firma kapsamı  : {good.ticker.nunique()} firma")
        size = sum(os.path.getsize(p) for p in good.path if p and os.path.exists(p))
        print(f"  Disk           : {size/1e9:.2f} GB (sıkıştırılmış)")
    print(f"\n  → {MANIFEST}")
    print("\n  SONRAKİ ADIM: özellik çıkarımı (Loughran-McDonald ve/veya FinBERT).")
    print("  Metin ham saklandığı için ikisi de yeniden indirme gerektirmez.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['cik', 'index', 'download', 'all'], default='all')
    ap.add_argument('--forms', nargs='+', default=['10-K', '10-Q', '8-K'],
                    help='listeye alınacak form tipleri (8-K item kodları için gerekli)')
    ap.add_argument('--download-forms', nargs='+', default=['10-K', '10-Q'],
                    help='METNİ indirilecek formlar. 8-K varsayılan olarak HARİÇ — '
                         'item kodları index aşamasında zaten elde ediliyor.')
    ap.add_argument('--rate', type=float, default=5.0, help='istek/saniye (SEC sınırı 10)')
    ap.add_argument('--limit', type=int, default=0, help='deneme için firma sayısı')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--user-agent', type=str, default=None,
                    help='SEC iletişim bilgisi, ör. "Ad Soyad eposta@ornek.com". '
                         'Verilmezse SEC_USER_AGENT ortam değişkeni kullanılır.')
    args = ap.parse_args()

    global _UA_OVERRIDE
    _UA_OVERRIDE = args.user_agent

    os.makedirs(TEXT_DIR, exist_ok=True)
    os.makedirs('datasets', exist_ok=True)

    if args.stage in ('cik', 'all'):
        if os.path.exists(CIK_MAP) and not args.overwrite and args.stage == 'all':
            print(f"[Atlandı] {CIK_MAP} mevcut (--overwrite ile yenile)\n")
        else:
            stage_cik()
    if args.stage in ('index', 'all'):
        if os.path.exists(INDEX) and not args.overwrite and args.stage == 'all':
            print(f"[Atlandı] {INDEX} mevcut\n")
        else:
            stage_index(args.forms, args.rate, args.limit)
    if args.stage in ('download', 'all'):
        stage_download(args.rate, args.limit, args.overwrite, args.download_forms)


if __name__ == '__main__':
    main()