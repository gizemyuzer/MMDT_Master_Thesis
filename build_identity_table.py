"""
build_identity_table.py — tek bir kimlik tablosu uretir.

NEDEN
    permno, lpermno/gvkey, cik, namedt/nameenddt su anda uc ayri WRDS
    tablosundan geliyor ve projede tek bir dosyada bulusmuyorlar. Uc kaynagin
    gecerlilik araliklari da birbirinden farkli: bir firma ismini degistirdiginde
    stocknames yeni satir aciyor, CCM baglantisi degistiginde linkhist yeni satir
    aciyor, CIK degistiginde wciklink yeni satir aciyor. Bu yuzden "her kolonu
    yan yana koymak" duz bir merge degil, ARALIK KESISIMI isi.

NE URETIR
    identity_intervals.csv
        Her satir bir (permno, baslangic, bitis) araligi ve o aralikta gecerli
        olan TUM kimlikler: ticker, comnam, shrcd, exchcd, siccd, gvkey, lpermco,
        liid, linktype, linkprim, cik, conm. Gunluk panele soyle baglanir:
            date >= start and date <= end
    identity_by_permno.csv
        PERMNO basina ozet: kac isim donemi, kac gvkey, kac cik, ilk/son tarih,
        borsadan cikis tarihi ve getirisi.
    identity_report.json
        Satir sayilari, eslesmeyen PERMNO'lar, bos kalan kolonlar.

PERMNO LISTESI
    datasets/universe.csv silinmisse, --from-evidence ile kohort
    results/.../candidate_interval_evidence.csv dosyasindan okunur. DIKKAT: o
    dosyada 398 PERMNO var, orijinal kohort 400. Eksik ikisi Compustat
    baglantisi bulunamayan firmalar; onlari geri getirmek icin universe.csv ya da
    fetch_fundamental_sources.py'nin rapor JSON'u gerekir.

KULLANIM
    python3 build_identity_table.py --universe datasets/universe.csv
    python3 build_identity_table.py --from-evidence results/archive_identity_.../candidate_interval_evidence.csv
    python3 build_identity_table.py --self-test      # WRDS gerektirmez
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

__build__ = "2026-09-25a"

START = "2009-01-01"
END = "2024-12-31"
OPEN_END = pd.Timestamp("2099-12-31")   # NULL / 9999-12-31 yerine

NAME_SQL = """
    SELECT permno, namedt, nameenddt, ticker, comnam, shrcd, exchcd, siccd
    FROM crsp.stocknames
    WHERE permno IN %(permnos)s
    ORDER BY permno, namedt
"""

LINK_SQL = """
    SELECT gvkey, lpermno, lpermco, liid, linktype, linkprim, linkdt, linkenddt
    FROM crsp.ccmxpf_lnkhist
    WHERE lpermno IN %(permnos)s
      AND linktype IN ('LC','LU')
      AND linkprim IN ('P','C')
    ORDER BY lpermno, linkdt
"""

CIK_SQL = """
    SELECT gvkey, cik, conm, datadate1 AS cikdt, datadate2 AS cikenddt
    FROM wrdssec.wciklink_gvkey
    WHERE gvkey IN %(gvkeys)s
    ORDER BY gvkey, datadate1
"""

DELIST_SQL = """
    SELECT permno, dlstdt, dlstcd, dlret
    FROM crsp.dsedelist
    WHERE permno IN %(permnos)s
    ORDER BY permno, dlstdt
"""


# ---------------------------------------------------------------- yardimcilar

def to_ts(s, default_end=False):
    """Tarihe cevir; bos ve 9999-12-31 degerleri acik ucu temsil eder."""
    t = pd.to_datetime(s, errors="coerce")
    if default_end:
        return t.fillna(OPEN_END).clip(upper=OPEN_END)
    return t


def normalise(frame, permno_col, start_col, end_col, keep):
    """Bir kaynagi ortak (permno, start, end, ...) semasina indirger."""
    f = frame.copy()
    f["permno"] = pd.to_numeric(f[permno_col], errors="raise").astype("int64")
    f["start"] = to_ts(f[start_col])
    f["end"] = to_ts(f[end_col], default_end=True)
    bad = f.start.isna()
    if bad.any():
        raise ValueError("Baslangic tarihi okunamayan %d satir" % int(bad.sum()))
    inverted = f.end < f.start
    if inverted.any():
        raise ValueError("Bitis tarihi baslangictan kucuk olan %d satir"
                         % int(inverted.sum()))
    return f[["permno", "start", "end"] + keep]


def intersect(sources, permnos):
    """Aralik kesisimi.

    Her PERMNO icin tum kaynaklardaki sinir tarihlerini toplayip zaman eksenini
    parcalara boler; her parca icinde her kaynaktan en fazla bir satir gecerli
    olur, dolayisiyla kolonlar yan yana yazilabilir. Kaynagin kapsamadigi
    parcalarda ilgili kolonlar bos kalir (ornegin Compustat baglantisi olmayan
    donemler).
    """
    out = []
    for permno in permnos:
        parts = {k: v.loc[v.permno == permno] for k, v in sources.items()}
        edges = {pd.Timestamp(START), pd.Timestamp(END) + pd.Timedelta(days=1)}
        for f in parts.values():
            for s, e in zip(f.start, f.end):
                edges.add(s)
                edges.add(e + pd.Timedelta(days=1))
        edges = sorted(t for t in edges
                       if pd.Timestamp(START) <= t <= pd.Timestamp(END) + pd.Timedelta(days=1))
        for a, b in zip(edges[:-1], edges[1:]):
            seg_end = b - pd.Timedelta(days=1)
            if seg_end < a:
                continue
            row = {"permno": permno, "start": a, "end": seg_end}
            empty = True
            for key, f in parts.items():
                hit = f.loc[(f.start <= a) & (f.end >= seg_end)]
                if len(hit) > 1:
                    # Ayni aralikta birden fazla gecerli satir: veri sorunu,
                    # sessizce ilkini almak yerine isaretle.
                    row["_conflict_" + key] = len(hit)
                    hit = hit.iloc[[0]]
                if len(hit) == 1:
                    empty = False
                    for c in hit.columns:
                        if c in ("permno", "start", "end"):
                            continue
                        row[c] = hit.iloc[0][c]
            if not empty:
                out.append(row)
    if not out:
        return pd.DataFrame(columns=["permno", "start", "end"])
    return pd.DataFrame(out).sort_values(["permno", "start"]).reset_index(drop=True)


def summarise(intervals, delist):
    rows = []
    for permno, g in intervals.groupby("permno"):
        rows.append({
            "permno": permno,
            "intervals": len(g),
            "name_periods": g.get("ticker", pd.Series(dtype=object)).nunique(dropna=True),
            "gvkeys": g.get("gvkey", pd.Series(dtype=object)).nunique(dropna=True),
            "ciks": g.get("cik", pd.Series(dtype=object)).nunique(dropna=True),
            "first_date": g.start.min(),
            "last_date": g.end.max(),
            "last_ticker": g.dropna(subset=["ticker"]).ticker.iloc[-1]
                           if "ticker" in g and g.ticker.notna().any() else None,
            "last_comnam": g.dropna(subset=["comnam"]).comnam.iloc[-1]
                           if "comnam" in g and g.comnam.notna().any() else None,
        })
    s = pd.DataFrame(rows)
    if len(delist):
        d = delist.sort_values("dlstdt").groupby("permno").tail(1)
        d = d.rename(columns={"dlstdt": "delist_date", "dlstcd": "delist_code",
                              "dlret": "delist_return"})
        s = s.merge(d[["permno", "delist_date", "delist_code", "delist_return"]],
                    on="permno", how="left")
    return s


# ---------------------------------------------------------------------- WRDS

def pull(permnos):
    import wrds
    db = wrds.Connection(wrds_username="gizemyuzer")
    try:
        p = tuple(int(x) for x in permnos)
        names = db.raw_sql(NAME_SQL, params={"permnos": p})
        links = db.raw_sql(LINK_SQL, params={"permnos": p})
        delist = db.raw_sql(DELIST_SQL, params={"permnos": p})
        gvkeys = tuple(sorted(set(links.gvkey.dropna().astype(str))))
        ciks = (db.raw_sql(CIK_SQL, params={"gvkeys": gvkeys})
                if gvkeys else pd.DataFrame(
                    columns=["gvkey", "cik", "conm", "cikdt", "cikenddt"]))
    finally:
        db.close()
    return names, links, delist, ciks


def build(names, links, delist, ciks, permnos):
    links = links.copy()
    links["gvkey"] = links.gvkey.astype(str).str.zfill(6)

    # CIK araliklari gvkey uzerinden permno'ya tasinir.
    if len(ciks):
        c = ciks.copy()
        c["gvkey"] = c.gvkey.astype(str).str.zfill(6)
        c = c.merge(links[["gvkey", "lpermno", "linkdt", "linkenddt"]],
                    on="gvkey", how="inner")
        # CIK araligi ile baglanti araliginin kesisimi
        c["s"] = np.maximum(to_ts(c.cikdt).values,
                            to_ts(c.linkdt).values)
        c["e"] = np.minimum(to_ts(c.cikenddt, True).values,
                            to_ts(c.linkenddt, True).values)
        c = c.loc[c.e >= c.s]
        cik_src = normalise(c, "lpermno", "s", "e", ["cik", "conm"])
    else:
        cik_src = pd.DataFrame(columns=["permno", "start", "end", "cik", "conm"])

    sources = {
        "name": normalise(names, "permno", "namedt", "nameenddt",
                          ["ticker", "comnam", "shrcd", "exchcd", "siccd"]),
        "link": normalise(links, "lpermno", "linkdt", "linkenddt",
                          ["gvkey", "lpermco", "liid", "linktype", "linkprim"]),
        "cik": cik_src,
    }
    intervals = intersect(sources, sorted(int(x) for x in permnos))
    order = ["permno", "start", "end", "ticker", "comnam", "shrcd", "exchcd",
             "siccd", "gvkey", "lpermco", "liid", "linktype", "linkprim",
             "cik", "conm"]
    for c in order:
        if c not in intervals.columns:
            intervals[c] = pd.NA
    extra = [c for c in intervals.columns if c not in order]
    intervals = intervals[order + extra]
    return intervals, summarise(intervals, delist)


# ----------------------------------------------------------------- self-test

def self_tests():
    # Iki isim donemi, tek baglanti, baglantinin ortasinda degisen CIK.
    names = pd.DataFrame({
        "permno": [10, 10],
        "namedt": ["2009-01-01", "2015-06-01"],
        "nameenddt": ["2015-05-31", None],
        "ticker": ["AAA", "BBB"], "comnam": ["Alpha", "Beta"],
        "shrcd": [11, 11], "exchcd": [1, 1], "siccd": [3711, 3711]})
    links = pd.DataFrame({
        "gvkey": ["1001"], "lpermno": [10], "lpermco": [500], "liid": ["01"],
        "linktype": ["LC"], "linkprim": ["P"],
        "linkdt": ["2009-01-01"], "linkenddt": [None]})
    ciks = pd.DataFrame({
        "gvkey": ["1001", "1001"], "cik": ["0000111", "0000222"],
        "conm": ["ALPHA INC", "BETA INC"],
        "datadate1": ["2009-01-01", "2018-01-01"],
        "datadate2": ["2017-12-31", None]}).rename(
            columns={"datadate1": "cikdt", "datadate2": "cikenddt"})
    delist = pd.DataFrame({"permno": [10], "dlstdt": ["2023-04-14"],
                           "dlstcd": [233], "dlret": [-0.0821]})

    iv, sm = build(names, links, delist, ciks, [10])

    # Uc sinir -> uc aralik: isim degisimi 2015-06-01, CIK degisimi 2018-01-01
    assert len(iv) == 3, iv
    assert list(iv.ticker) == ["AAA", "BBB", "BBB"]
    assert list(iv.cik) == ["0000111", "0000111", "0000222"]
    assert iv.gvkey.eq("001001").all()
    assert str(iv.start.iloc[1].date()) == "2015-06-01"
    assert str(iv.start.iloc[2].date()) == "2018-01-01"
    assert str(iv.end.iloc[-1].date()) == END
    assert sm.loc[0, "gvkeys"] == 1 and sm.loc[0, "ciks"] == 2
    assert sm.loc[0, "delist_code"] == 233

    # Baglantisi olmayan donem: kolonlar bos kalir, satir yine uretilir.
    links2 = links.copy()
    links2["linkenddt"] = ["2012-12-31"]
    iv2, _ = build(names, links2, delist, ciks, [10])
    assert iv2.gvkey.isna().any() and iv2.ticker.notna().all()

    # Kesisim bos kalmamali: her gun tam olarak bir aralikta olmali.
    days = pd.date_range(START, END, freq="D")
    cover = np.zeros(len(days), dtype=int)
    for s, e in zip(iv.start, iv.end):
        cover += ((days >= s) & (days <= e)).astype(int)
    assert (cover == 1).all(), "araliklar ortusuyor veya bosluk var"

    print("self-tests OK")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", help="datasets/universe.csv")
    ap.add_argument("--from-evidence",
                    help="results/.../candidate_interval_evidence.csv (398 PERMNO)")
    ap.add_argument("--outdir", default="datasets")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        self_tests()
        return

    if a.universe:
        u = pd.read_csv(a.universe)
        if "permno" not in u.columns:
            sys.exit("universe.csv icinde permno kolonu yok")
        permnos = sorted(set(pd.to_numeric(u.permno, errors="raise").astype(int)))
        source = a.universe
    elif a.from_evidence:
        e = pd.read_csv(a.from_evidence)
        permnos = sorted(set(pd.to_numeric(e.lpermno, errors="raise").astype(int)))
        source = a.from_evidence
    else:
        sys.exit("--universe ya da --from-evidence ver")

    print("PERMNO sayisi: %d (%s)" % (len(permnos), source))
    if len(permnos) != 400:
        print("UYARI: orijinal kohort 400 PERMNO; %d okundu." % len(permnos))

    self_tests()
    names, links, delist, ciks = pull(permnos)
    intervals, summary = build(names, links, delist, ciks, permnos)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.outdir) / ("identity_%s_%s" % (stamp, uuid4().hex[:8]))
    out.mkdir(parents=True, exist_ok=False)
    intervals.to_csv(out / "identity_intervals.csv", index=False)
    summary.to_csv(out / "identity_by_permno.csv", index=False)
    names.to_csv(out / "name_history.csv", index=False)
    links.to_csv(out / "ccm_links.csv", index=False)
    ciks.to_csv(out / "cik_links.csv", index=False)
    delist.to_csv(out / "delistings.csv", index=False)

    missing = {
        "no_name_history": sorted(set(permnos) - set(names.permno.astype(int))),
        "no_ccm_link": sorted(set(permnos) - set(links.lpermno.astype(int))),
        "no_cik": sorted(set(permnos) - set(
            intervals.loc[intervals.cik.notna(), "permno"].astype(int))),
    }
    report = {
        "build": __build__,
        "source": source,
        "permnos_requested": len(permnos),
        "interval_rows": len(intervals),
        "permnos_with_intervals": int(intervals.permno.nunique()),
        "null_share": {c: round(float(intervals[c].isna().mean()), 4)
                       for c in intervals.columns},
        "conflicts": {c: int(intervals[c].notna().sum())
                      for c in intervals.columns if c.startswith("_conflict_")},
        **{k: {"count": len(v), "permnos": v[:20]} for k, v in missing.items()},
    }
    (out / "identity_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("interval_rows", "permnos_with_intervals",
                       "no_name_history", "no_ccm_link", "no_cik")
                      if k in report}, indent=2, default=str))
    print("->", out)


if __name__ == "__main__":
    main()