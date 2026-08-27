"""
apply_patch.py — modalite IV (metin) degisikliklerini otomatik uygular.

Idempotent: iki kez calistirirsaniz ikinci sefer hicbir sey yapmaz.
Her dosyanin .bak yedegini alir.

KULLANIM:
    python apply_patch.py --dry-run     # once ne yapacagini goster
    python apply_patch.py               # uygula
"""
import os
import shutil
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))

FE = os.path.join(HERE, 'datasets', 'feature_engineering.py')
XGB = os.path.join(HERE, 'run_xgb_factorial.py')

# ── Blok 1: TEXT kolon tanimlari ──
BLOCK_TEXT_COLS = '''
# ═══════════════════════════════════════════════════════════════════
# MODALİTE IV — METİN (SEC EDGAR)
# ═══════════════════════════════════════════════════════════════════
# build_text_features.py üretir, attach_text_features() panele bağlar.
# text_event : 8-K item kodlarından olay sayımları (günlük, NLP yok)
# text_lm    : 10-K/10-Q metninden LM oranları + benzerlik (çeyreklik)
# Ayrı tutulması, metin işe yaramazsa hangi katmanın sorumlu olduğunu
# ayırt etmeyi sağlar.
TEXT_EVENT_COLS = [
    'EK_Bankruptcy', 'EK_DebtTrigger', 'EK_Impairment', 'EK_ListingWarn',
    'EK_AuditorChange', 'EK_Restatement', 'EK_MgmtChange',
    'EK_Activity_60d', 'EK_DaysSince',
]
TEXT_LM_COLS = [
    'LM_Negative', 'LM_Positive', 'LM_Uncertainty', 'LM_Litigious',
    'LM_Constraining', 'LM_WordCount',
    'TXT_SimPrev', 'TXT_LenChange', 'TXT_DaysSinceFiling',
]
# 400 firmanın 6'sının CIK'i bulunamadı; onlarda olay sayımını 0 yapmak
# "olay yok" demek olurdu, oysa doğrusu "bilinmiyor".
TEXT_COVERAGE_COL = 'TXT_HasCoverage'
TEXT_COLS = TEXT_EVENT_COLS + TEXT_LM_COLS + [TEXT_COVERAGE_COL]
'''

# ── Blok 2: FEATURE_GROUPS eklemeleri ──
ANCHOR_GROUPS = "    'interact': INTERACTION_COLS,"
BLOCK_GROUPS = """    'interact': INTERACTION_COLS,
    # Modalite IV
    'text': TEXT_COLS,
    'text_event': TEXT_EVENT_COLS + [TEXT_COVERAGE_COL],
    'text_lm': TEXT_LM_COLS + [TEXT_COVERAGE_COL],"""

# ── Blok 3: attach_text_features fonksiyonu ──
BLOCK_ATTACH = '''def attach_text_features(df, path=os.path.join('datasets', 'text_features.csv')):
    """
    Modalite IV'ü panele bağlar. build_text_features.py çıktısını okur.

    CACHE'E YAZILMAZ — bilinçli. Metin özellikleri gelişiyor (LM → FinBERT);
    cache'e gömülse dosya yeniden üretildiğinde eski değerler sessizce
    kullanılmaya devam ederdi.

    EKSİK DEĞER MANTIĞI — iki farklı "yok" ayırt edilir:
      · Olay kolonları (8-K): NaN → 0. O tarihe kadar 8-K yoksa "olay
        olmadı" doğru okumadır.
      · LM kolonları: NaN BIRAKILIR. İlk 10-K yayımlanmadan önce duygu
        bilinmez; sıfır yazmak "nötr duygu" demek olurdu.
      · TXT_HasCoverage: model "olay yok" ile "veri yok"u ayırt edebilsin.
    """
    if not os.path.exists(path):
        print(f"  [Metin] {path} yok — modalite IV atlanıyor.")
        print(f"          Üretmek için: python build_text_features.py")
        return df

    t = pd.read_csv(path)
    if 'ticker' in t.columns and 'Ticker' not in t.columns:
        t = t.rename(columns={'ticker': 'Ticker'})
    t['date'] = pd.to_datetime(t['date'])
    t = t.drop_duplicates(subset=['date', 'Ticker'], keep='last')

    feat_cols = [c for c in t.columns if c not in ('date', 'Ticker')]
    if not feat_cols:
        print(f"  [Metin] {path} özellik kolonu içermiyor — atlanıyor.")
        return df

    # Pozisyonel birleştirme: indeks hizalaması devreye girmesin
    left = pd.DataFrame({'date': pd.to_datetime(df.index.values),
                         'Ticker': df['Ticker'].values})
    merged = left.merge(t[['date', 'Ticker'] + feat_cols],
                        on=['date', 'Ticker'], how='left')

    covered = set(t['Ticker'].dropna().unique())
    df[TEXT_COVERAGE_COL] = df['Ticker'].isin(covered).astype(float).values

    for c in feat_cols:
        vals = merged[c].values
        if c in TEXT_EVENT_COLS:
            vals = pd.Series(vals).fillna(0.0).values
        df[c] = vals

    n_firms = df.loc[df[TEXT_COVERAGE_COL] > 0, 'Ticker'].nunique()
    tot_firms = df['Ticker'].nunique()
    lm_present = [c for c in TEXT_LM_COLS if c in df.columns]
    ev_present = [c for c in TEXT_EVENT_COLS if c in df.columns]
    print(f"  [Metin] Modalite IV bağlandı: {len(ev_present)} olay + "
          f"{len(lm_present)} LM kolonu")
    print(f"          Kapsama: {n_firms}/{tot_firms} firma "
          f"(%{100 * n_firms / max(tot_firms, 1):.1f})")
    if lm_present:
        nn = df[lm_present[0]].notna().mean()
        print(f"          LM kolonları dolu satır oranı: %{100 * nn:.1f}")
    return df


'''

BLOCK_XGB = """    'I+II-xs':  (('tech', 'fund', 'fund_xs'),  'Teknik + firma + kesitsel'),
    # ── Modalite IV: metin (SEC EDGAR) ──
    'IV':        (('text',),                       'Sadece metin'),
    'I+II+IV':   (('tech', 'fund', 'text'),        'Teknik + firma + metin'),
    'I+II+IV-e': (('tech', 'fund', 'text_event'),  'Teknik + firma + 8-K olayları'),
    'I+II+IV-l': (('tech', 'fund', 'text_lm'),     'Teknik + firma + LM duygu'),"""


def patch(path, edits, dry):
    """edits: [(ad, kontrol_metni, eski, yeni)]"""
    if not os.path.exists(path):
        print(f"  DOSYA YOK: {path}")
        return False
    with open(path, encoding='utf-8') as f:
        src = f.read()
    orig = src
    for name, marker, old, new in edits:
        if marker in src:
            print(f"  ATLANDI  {name} (zaten var)")
            continue
        if old not in src:
            print(f"  BULUNAMADI  {name} — capa metni yok, elle bakin")
            continue
        if src.count(old) > 1:
            print(f"  BELIRSIZ  {name} — capa {src.count(old)} kez geciyor, elle bakin")
            continue
        src = src.replace(old, new, 1)
        print(f"  UYGULANDI  {name}")
    if src == orig:
        return False
    if dry:
        print("  (dry-run: dosya yazilmadi)")
        return True
    shutil.copy2(path, path + '.bak')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(src)
    print(f"  YAZILDI  {path}  (yedek: {os.path.basename(path)}.bak)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    anchor_xs = 'FIRM_FUNDAMENTAL_XS_COLS = [c + XS_SUFFIX for c in FIRM_FUNDAMENTAL_COLS]'

    print(f"\n### {FE}")
    patch(FE, [
        ('TEXT kolon tanimlari', 'TEXT_EVENT_COLS',
         anchor_xs, anchor_xs + '\n' + BLOCK_TEXT_COLS),
        ('FEATURE_GROUPS eklemeleri', "'text_event':",
         ANCHOR_GROUPS, BLOCK_GROUPS),
        ('attach_text_features', 'def attach_text_features',
         'def prepare_dataset(force_refresh=False):',
         BLOCK_ATTACH + 'def prepare_dataset(force_refresh=False):'),
        ('cache dalinda cagri', 'attach_text_features(dataset_out)',
         '        return dataset_out',
         '        dataset_out = attach_text_features(dataset_out)\n        return dataset_out'),
    ], args.dry_run)

    # Son cagri ayri: yukaridaki adim 'attach_text_features(dataset_out)'
    # isaretini olusturdugu icin ikinci cagri atlanir. Ayri kontrol metni:
    with open(FE, encoding='utf-8') as f:
        s = f.read()
    if s.count('attach_text_features(dataset_out)') < 2:
        patch(FE, [('fonksiyon sonu cagri', 'ZZZ_YOK_MARKER',
                    '\n    return dataset_out\n',
                    '\n    dataset_out = attach_text_features(dataset_out)\n    return dataset_out\n')],
              args.dry_run)

    print(f"\n### {XGB}")
    patch(XGB, [
        ('XGB metin hucreleri', "'I+II+IV'",
         "    'I+II-xs':  (('tech', 'fund', 'fund_xs'),  'Teknik + firma + kesitsel'),",
         BLOCK_XGB),
    ], args.dry_run)

    print("\nBitti. Kontrol: python check_state.py")


if __name__ == '__main__':
    main()