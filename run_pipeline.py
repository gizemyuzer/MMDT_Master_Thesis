"""
verify_pipeline.py
───────────────────
BİLİNEN-CEVAP TESTLERİ (known-answer tests).

═══════════════════════════════════════════════════════════════════════
NEDEN
═══════════════════════════════════════════════════════════════════════
Portföy simülasyonunda bir yön hatası bulundu: sıralama tabanlı seçimde
"en riskliyi çıkar" yerine "en güvenliyi çıkar" yapılıyordu. Hata sessizdi —
kod çalıştı, makul görünen sayılar üretti, ve sonuç yanlış yorumlandı.

Bu tür hatalara karşı tek gerçek savunma, cevabı ÖNCEDEN BİLİNEN girdilerle
tüm hattı sınamaktır. Mükemmel bir tahminci MCC=+1 vermeli; tersine çevrilmiş
bir tahminci MCC=−1; rastgele bir tahminci ≈0. Backtest tarafında da mükemmel
tahminci en yüksek, ters tahminci en düşük performansı vermeli.

Bu script bunları test eder. Hepsi geçerse, değerlendirme ve backtest
hatlarının yönü doğrudur — "sanırım doğru" yerine "test ettim" diyebilirsiniz.

Tezin metodoloji bölümüne tek cümleyle girer:
    "Değerlendirme ve backtest hatları, cevabı analitik olarak bilinen
     yapay tahminciler (mükemmel / rastgele / tersine çevrilmiş) ile
     doğrulanmıştır."

KULLANIM:
    python verify_pipeline.py
Çıkış kodu 0 = tüm testler geçti, 1 = en az bir test başarısız.
"""
import sys

import numpy as np
import pandas as pd

PASS, FAIL = "✓ GEÇTİ", "✗ BAŞARISIZ"
results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition), detail))
    print(f"  {PASS if condition else FAIL}  {name}")
    if detail:
        print(f"           {detail}")
    return bool(condition)


# ══════════════════════════════════════════════════════════════════
# 1) SINIFLANDIRMA METRİKLERİ
# ══════════════════════════════════════════════════════════════════
def test_classification():
    print("\n[1] Sınıflandırma metrikleri — bilinen cevaplar")
    from models.pytorch_trainer import (_compute_classification_metrics,
                                        find_best_threshold_mcc)
    rng = np.random.default_rng(0)
    n = 20_000
    y = (rng.random(n) < 0.13).astype(int)      # %13 taban oran (gerçeğe yakın)

    # (a) Mükemmel tahminci → MCC = +1, ROC-AUC = 1
    p_perfect = y.astype(float) * 0.9 + 0.05
    m = _compute_classification_metrics(p_perfect, y, 0.5)
    check("mükemmel tahminci → MCC ≈ +1", m['mcc'] > 0.99,
          f"MCC={m['mcc']:.4f}  ROC-AUC={m['roc_auc']:.4f}")
    check("mükemmel tahminci → ROC-AUC ≈ 1", m['roc_auc'] > 0.99)

    # (b) Tersine çevrilmiş tahminci → MCC = −1, ROC-AUC = 0
    p_inv = 1.0 - p_perfect
    m = _compute_classification_metrics(p_inv, y, 0.5)
    check("ters tahminci → MCC ≈ −1", m['mcc'] < -0.99,
          f"MCC={m['mcc']:.4f}  ROC-AUC={m['roc_auc']:.4f}")
    check("ters tahminci → ROC-AUC ≈ 0", m['roc_auc'] < 0.01)

    # (c) Rastgele tahminci → MCC ≈ 0, ROC-AUC ≈ 0.5
    p_rand = rng.random(n)
    m = _compute_classification_metrics(p_rand, y, 0.5)
    check("rastgele tahminci → MCC ≈ 0", abs(m['mcc']) < 0.03,
          f"MCC={m['mcc']:.4f}  ROC-AUC={m['roc_auc']:.4f}")
    check("rastgele tahminci → ROC-AUC ≈ 0.5", abs(m['roc_auc'] - 0.5) < 0.03)

    # (d) PR-AUC taban oranı: rastgele tahmincide ≈ pozitif oran
    check("rastgele PR-AUC ≈ taban oran", abs(m['pr_auc'] - y.mean()) < 0.02,
          f"PR-AUC={m['pr_auc']:.4f}  taban oran={y.mean():.4f}")

    # (e) Eşik seçimi: sinyalli veride pozitif MCC bulmalı
    p_signal = np.clip(y * 0.35 + rng.normal(0.3, 0.15, n), 0, 1)
    thr, best = find_best_threshold_mcc(y, p_signal)
    check("eşik seçimi sinyali buluyor", best > 0.1,
          f"eşik={thr:.3f}  val MCC={best:.4f}")

    # (f) Dejenere koruma: tek sınıf tahmininde MCC=0 dönmeli
    thr2, best2 = find_best_threshold_mcc(np.zeros(100, dtype=int), rng.random(100))
    check("tek sınıflı hedefte dejenere korunuyor", best2 == 0.0)


# ══════════════════════════════════════════════════════════════════
# 2) BACKTEST YÖNÜ — sıralama tabanlı seçim
# ══════════════════════════════════════════════════════════════════
def test_backtest_direction():
    print("\n[2] Portföy simülasyonu — yön testi")
    src = open('run_portfolio_simulation.py', encoding='utf-8').read()
    ns = {'np': np, 'pd': pd}
    exec(src[src.index('def simulate('):src.index('def main(')], ns)
    simulate = ns['simulate']

    dates = pd.bdate_range('2022-01-01', periods=200)
    tick = [f'S{i}' for i in range(20)]
    ret = pd.DataFrame(0.0, index=dates, columns=tick)
    ret[tick[:10]] = -0.003          # riskli grup: kaybettiriyor
    ret[tick[10:]] = +0.003          # güvenli grup: kazandırıyor
    sc = pd.DataFrame(0.0, index=dates, columns=tick)
    sc[tick[:10]] = 0.9              # yüksek skor = riskli
    sc[tick[10:]] = 0.1
    rebal = set(dates[::20])

    net, _, _ = simulate(sc, ret, rebal, 50.0, 0.0)
    total = (1 + net).prod() - 1
    check("en riskli %50 çıkarılınca getiri POZİTİF", total > 0,
          f"toplam getiri = {total*100:+.2f}%  (kaybettiren grup çıkarıldı)")

    # Hiç çıkarma yok → getiri ≈ 0 (iki grup birbirini götürür)
    net0, _, _ = simulate(sc, ret, rebal, 0.0, 0.0)
    tot0 = (1 + net0).prod() - 1
    check("hiç çıkarma yoksa getiri ≈ 0", abs(tot0) < 0.02,
          f"toplam getiri = {tot0*100:+.2f}%")

    # Koruma mekanizması: yön ters verilirse hata fırlatmalı
    try:
        simulate(sc, ret, rebal, 50.0, 0.0, higher_is_riskier=False)
        check("yön koruması hatayı yakalıyor", False, "RuntimeError beklenirdi")
    except RuntimeError:
        check("yön koruması hatayı yakalıyor", True)


# ══════════════════════════════════════════════════════════════════
# 3) EŞİK TABANLI BACKTEST YÖNÜ (regime analizi)
# ══════════════════════════════════════════════════════════════════
def test_threshold_backtest():
    print("\n[3] Eşik tabanlı backtest (rejim analizi) — yön testi")
    dates = pd.bdate_range('2022-01-01', periods=200)
    tick = [f'S{i}' for i in range(20)]
    ret = pd.DataFrame(0.0, index=dates, columns=tick)
    ret[tick[:10]] = -0.003
    ret[tick[10:]] = +0.003
    prob = pd.DataFrame(0.0, index=dates, columns=tick)
    prob[tick[:10]] = 0.9
    prob[tick[10:]] = 0.1

    # run_regime_analysis.py / run_economic_value_backtest.py ile aynı mantık
    flagged = prob >= 0.5
    n_safe = (~flagged).sum(axis=1).replace(0, np.nan)
    w = (~flagged).astype(float).div(n_safe, axis=0).fillna(0.0)
    strat = (w.shift(1).fillna(0.0) * ret).sum(axis=1)
    total = (1 + strat).prod() - 1
    check("(~flagged) mantığı doğru yönde", total > 0,
          f"toplam getiri = {total*100:+.2f}%")


# ══════════════════════════════════════════════════════════════════
# 4) PERFORMANS METRİKLERİ
# ══════════════════════════════════════════════════════════════════
def test_performance_metrics():
    print("\n[4] Performans metrikleri — bilinen cevaplar")
    src = open('run_portfolio_simulation.py', encoding='utf-8').read()
    ns = {'np': np, 'pd': pd, 'TRADING_DAYS': 252}
    exec(src[src.index('def performance('):src.index('def simulate(')], ns)
    performance = ns['performance']

    dates = pd.bdate_range('2022-01-01', periods=252)
    # Sabit +%0,1/gün → drawdown olmamalı, getiri ≈ e^(252*0.001)-1
    r = pd.Series(0.001, index=dates)
    m = performance(r, 'test')
    check("sürekli artan seride maxDD ≈ 0", abs(m['max_drawdown']) < 1e-6,
          f"maxDD={m['max_drawdown']:.6f}")
    expected = (1.001 ** 252) - 1
    check("toplam getiri doğru hesaplanıyor", abs(m['total_return'] - expected) < 1e-6,
          f"hesaplanan={m['total_return']:.4f}  beklenen={expected:.4f}")

    # %50 düşüş sonrası toparlanma → maxDD ≈ −50%
    r2 = pd.Series(0.0, index=dates)
    r2.iloc[10] = -0.5
    r2.iloc[11:] = 0.002
    m2 = performance(r2, 'test')
    check("maxDD ≈ −50% yakalanıyor", abs(m2['max_drawdown'] + 0.5) < 0.01,
          f"maxDD={m2['max_drawdown']*100:.2f}%")


# ══════════════════════════════════════════════════════════════════
# 5) ZAMANLAMA — ileriye bakış yok
# ══════════════════════════════════════════════════════════════════
def test_no_lookahead():
    print("\n[5] Zamanlama — ileriye bakış kontrolü")
    src = open('run_portfolio_simulation.py', encoding='utf-8').read()
    ns = {'np': np, 'pd': pd}
    exec(src[src.index('def simulate('):src.index('def main(')], ns)
    simulate = ns['simulate']

    dates = pd.bdate_range('2022-01-01', periods=100)
    tick = [f'S{i}' for i in range(20)]
    ret = pd.DataFrame(0.001, index=dates, columns=tick)
    # Skorlar AYRIK (beraberlik yok) — aksi halde sıralama keyfi olur ve
    # test ölçmek istediği şeyi ölçmez.
    sc = pd.DataFrame(np.linspace(0.05, 0.95, 20)[None, :].repeat(len(dates), 0),
                      index=dates, columns=tick)
    rebal = set(dates[::20])

    # (a) Ağırlıklar shift(1) ile uygulanıyor mu → ilk gün pozisyon yok
    net, _, _ = simulate(sc, ret, rebal, 50.0, 0.0)
    check("ilk günde pozisyon yok (shift uygulanmış)", abs(net.iloc[0]) < 1e-12,
          f"ilk gün getirisi = {net.iloc[0]:.2e}")

    # (b) ASIL İLERİYE BAKIŞ TESTİ:
    #     Son rebalance'tan SONRAKİ bir getiriyi değiştirelim. Eğer kod
    #     ileriye bakmıyorsa, o tarihten ÖNCEKİ portföy getirileri hiç
    #     değişmemeli. Değişiyorsa gelecekten bilgi sızıyor demektir.
    ret2 = ret.copy()
    ret2.iloc[-1, :] = 5.0            # son günde uç bir şok
    net2, _, _ = simulate(sc, ret2, rebal, 50.0, 0.0)
    before = net.iloc[:-1]
    before2 = net2.iloc[:-1]
    check("gelecekteki getiri, geçmiş portföyü etkilemiyor",
          np.allclose(before.values, before2.values, atol=1e-12),
          f"son gün öncesi maks. fark = {np.abs(before.values-before2.values).max():.2e}")

    # (c) Skorlardaki gelecek bilgisi de sızmamalı: son rebalance sonrası
    #     skorları değiştirmek, önceki ağırlıkları etkilememeli
    sc2 = sc.copy()
    last_rb = max(d for d in rebal)
    sc2.loc[sc2.index > last_rb] = 0.99
    net3, _, _ = simulate(sc2, ret, rebal, 50.0, 0.0)
    check("rebalance sonrası skor değişimi geçmişi etkilemiyor",
          np.allclose(net.values, net3.values, atol=1e-12),
          f"maks. fark = {np.abs(net.values-net3.values).max():.2e}")


# ══════════════════════════════════════════════════════════════════
def main():
    print("═" * 70)
    print("BİLİNEN-CEVAP TESTLERİ")
    print("═" * 70)
    for fn in (test_classification, test_backtest_direction,
               test_threshold_backtest, test_performance_metrics,
               test_no_lookahead):
        try:
            fn()
        except Exception as e:
            print(f"  {FAIL}  {fn.__name__} çalıştırılamadı: {e}")
            import traceback
            traceback.print_exc()
            results.append((fn.__name__, False, str(e)))

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    print("\n" + "═" * 70)
    print(f"SONUÇ: {n_pass} geçti, {n_fail} başarısız ({len(results)} test)")
    print("═" * 70)
    if n_fail:
        print("\nBaşarısız testler:")
        for name, ok, detail in results:
            if not ok:
                print(f"  ✗ {name}  {detail}")
        print("\n⚠️ Sonuçları raporlamadan önce bunları düzeltin.")
    else:
        print("\nDeğerlendirme ve backtest hatlarının yönü doğrulandı.")
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())