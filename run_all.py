"""
run_all.py
───────────
Üç aşamayı gözetimsiz olarak arka arkaya çalıştırır:
  1) run_core_rerun.py       — XGBoost, tech_only, fund_only, multimodal
  2) run_multiseed.py        — concat, gated_cross_attention, static_context, film
  3) run_late_fusion.py      — tech/fund ayrı eğitim + logistic stacking

TASARIM NOTLARI (gece boyu koşacağı için):
  - Her aşama AYRI bir subprocess'te çalışır. Biri çökerse diğerleri yine de
    koşar; tek bir hata bütün geceyi çöpe atmaz.
  - Tüm çıktı hem ekrana hem results/logs/<aşama>.log dosyasına yazılır.
    Sabah hangi aşamada ne olduğunu log'dan okuyabilirsiniz.
  - Hiçbir aşama input() ile soru sormaz (--yes / --force verilir), aksi
    halde script gece boyunca bir soruda asılı kalırdı.
  - PYTHONUNBUFFERED=1 → log dosyası anlık dolar, çökerse bile son satırlar
    kaybolmaz (aksi halde tampon boşalmadan ölürse log boş görünür).
  - Sonunda özet tablo basılır: hangi aşama başarılı, kaç dakika sürdü.

ÖNKOŞUL: datasets/final_dataset.csv güncel ve fundamental kolonları dolu
olmalı (python check_fund_data.py ile doğrulanabilir — NaN% ~%19 olmalı,
%100 DEĞİL).

KULLANIM:
    python run_all.py
"""
import os
import subprocess
import sys
import time
from datetime import datetime

LOG_DIR = os.path.join('results', 'logs')

# (isim, komut) — sıra önemli, yukarıdan aşağı çalışır
STAGES = [
    ('1_core_rerun', [sys.executable, 'run_core_rerun.py', '--yes']),
    ('2_multiseed', [sys.executable, 'run_multiseed.py',
                     '--fusions', 'concat', 'gated_cross_attention',
                     'static_context', 'film', '--force']),
    ('3_late_fusion', [sys.executable, 'run_late_fusion.py', '--force']),
]


def run_stage(name, cmd):
    """Tek bir aşamayı çalıştırır, çıktıyı ekrana + log dosyasına yazar."""
    log_path = os.path.join(LOG_DIR, f'{name}.log')
    print("\n" + "█" * 78)
    print(f"AŞAMA: {name}")
    print(f"Komut: {' '.join(cmd)}")
    print(f"Log  : {log_path}")
    print(f"Başlangıç: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("█" * 78 + "\n", flush=True)

    # Alt süreçlerin çıktısını satır satır al — tamponlanıp kaybolmasın.
    #
    # PYTHONIOENCODING='utf-8' ZORUNLU (Windows):
    # Çıktı bir terminale değil pipe'a yazıldığında Python, Windows'ta
    # varsayılan kod sayfasına (cp1252) düşer. Script'lerdeki '→', '✓', 'ü'
    # gibi karakterler cp1252'de yok → UnicodeEncodeError ile daha ilk
    # print()'te çöker (kodda hiçbir mantık hatası olmadan). utf-8'e
    # sabitlemek bunu tamamen ortadan kaldırır.
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')

    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8', errors='replace') as logf:
        logf.write(f"=== {name} | başlangıç {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1, env=env,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        proc.wait()
        elapsed = (time.time() - t0) / 60
        logf.write(f"\n=== {name} | bitiş {datetime.now():%Y-%m-%d %H:%M:%S} | "
                   f"exit={proc.returncode} | {elapsed:.1f} dk ===\n")

    ok = (proc.returncode == 0)
    status = "✓ BAŞARILI" if ok else f"✗ HATA (exit={proc.returncode})"
    print(f"\n{status} — {name} | {elapsed:.1f} dk", flush=True)
    return {'stage': name, 'ok': ok, 'minutes': round(elapsed, 1),
            'exit_code': proc.returncode, 'log': log_path}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=str, nargs='+', default=None,
                    help="Sadece belirtilen aşamaları koştur, ör: "
                         "--only 2_multiseed 3_late_fusion "
                         "(1. aşama zaten tamamlandıysa 66 dk'lık Optuna'yı "
                         "boşuna tekrarlamamak için)")
    args = ap.parse_args()

    stages = STAGES if args.only is None else [s for s in STAGES if s[0] in args.only]
    if not stages:
        print(f"⚠️ '{args.only}' hiçbir aşamayla eşleşmedi. "
              f"Geçerli isimler: {[s[0] for s in STAGES]}")
        return 1

    os.makedirs(LOG_DIR, exist_ok=True)

    print("═" * 78)
    print("RUN ALL — üç aşama gözetimsiz olarak arka arkaya çalışacak")
    print("═" * 78)
    print(f"  Başlangıç: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Aşamalar : {[s[0] for s in stages]}")
    print(f"  Log klasörü: {LOG_DIR}")
    print("  NOT: Bir aşama çökerse diğerleri yine de çalışır.")

    results = []
    t_start = time.time()
    for name, cmd in stages:
        try:
            results.append(run_stage(name, cmd))
        except Exception as e:
            print(f"\n✗ {name} başlatılamadı: {e}", flush=True)
            results.append({'stage': name, 'ok': False, 'minutes': 0.0,
                            'exit_code': -1, 'log': '-'})

    total = (time.time() - t_start) / 60

    print("\n" + "═" * 78)
    print("ÖZET")
    print("═" * 78)
    for r in results:
        mark = "✓" if r['ok'] else "✗"
        print(f"  {mark} {r['stage']:<18} {r['minutes']:>7.1f} dk   log: {r['log']}")
    print(f"\n  Toplam süre: {total:.1f} dk ({total / 60:.1f} saat)")
    print(f"  Bitiş: {datetime.now():%Y-%m-%d %H:%M:%S}")

    print("\n  Sonuç dosyaları:")
    for p in ['results/core_rerun_summary.csv',
              'results/multiseed_summary.csv',
              'results/late_fusion_summary.csv']:
        exists = "var" if os.path.exists(p) else "YOK"
        print(f"    {p:<40} [{exists}]")

    n_fail = sum(1 for r in results if not r['ok'])
    if n_fail:
        print(f"\n  ⚠️ {n_fail} aşama başarısız — ilgili .log dosyasına bakın.")
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())