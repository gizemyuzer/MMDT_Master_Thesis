"""
check_build.py — hangi sürümün çalıştığını tek komutla söyler.

Mac'te düzenlenen dosyalar Windows'a kopyalanmadığında, koşu sessizce
eski kodla yapılıyor ve sonuç "düzeltme işe yaramadı" gibi görünüyor.
Bu script her dosyanın build damgasını ve sha256'sını basar; iki
makinede aynı çıktıyı görüyorsan senkronsun.

KULLANIM
    python check_build.py
"""
import hashlib, os, re, sys

BEKLENEN = {
    'portfolio_engine.py':  '2026-09-12b',
    'run_portfolio_v2.py':  '2026-09-12b',
    'run_mechanism_2x2.py': '2026-09-12b',
    'run_economic_all.py':  '2026-09-12a',
    'export_preds.py':      None,
    'run_common_sample.py': None,
}

print(f"{'dosya':<26}{'build':<14}{'sha256':<12}{'durum'}")
print('-' * 62)
sorun = 0
for f, want in BEKLENEN.items():
    if not os.path.exists(f):
        print(f"{f:<26}{'—':<14}{'—':<12}DOSYA YOK"); sorun += 1; continue
    raw = open(f, 'rb').read()
    h = hashlib.sha256(raw).hexdigest()[:8]
    m = re.search(rb'__build__ = "([^"]*)"', raw)
    got = m.group(1).decode() if m else '(damga yok)'
    if want is None:
        durum = '—'
    elif got == want:
        durum = 'GÜNCEL'
    else:
        durum = f'ESKİ! beklenen {want}'; sorun += 1
    print(f"{f:<26}{got:<14}{h:<12}{durum}")
print('-' * 62)
if sorun:
    print(f"\n⚠️  {sorun} dosya güncel değil. Mac'teki hâllerini kopyala,")
    print("   sonra bu scripti tekrar çalıştır.")
    sys.exit(1)
print("\n✓ Tüm dosyalar güncel.")