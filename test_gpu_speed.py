"""
test_gpu_speed.py — GPU bellek + hiz testi (2-3 dakika)

Gercek modeli gercek boyutlarla kurar, ileri+geri gecis yapar, tepe bellek
ve verimi olcer. Sonra tam bir egitim kosusunun ne kadar surecegini tahmin
eder.

Neden gerekli: GPU %100 kullanimda oldugunda is "sigar ama surunur" haline
gelebilir. Bunu 3 dakikada ogrenmek, 20 saat bekleyip ogrenmekten iyidir.

KULLANIM:
    python test_gpu_speed.py
    python test_gpu_speed.py --fund-dim 16      # metin hucresi icin
"""
import argparse
import time

import torch

from models.transformer_model import DualEncoderTransformer
from models.losses import FocalLoss


def mb(x):
    return x / 1024 ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', type=int, nargs='+', default=[32, 64, 128])
    ap.add_argument('--tech-dim', type=int, default=32)
    ap.add_argument('--fund-dim', type=int, default=6)
    ap.add_argument('--seq-len', type=int, default=20)
    ap.add_argument('--n-train', type=int, default=750_000,
                    help='egitim dizisi sayisi (bear market icin ~870000)')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--steps', type=int, default=30, help='olcum adimi')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA yok.")

    dev = torch.device('cuda')
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()

    print("=" * 66)
    print("GPU DURUMU")
    print("=" * 66)
    print(f"  Cihaz : {props.name}")
    print(f"  Toplam: {mb(total):,.0f} MB")
    print(f"  Bos   : {mb(free):,.0f} MB")
    print(f"  Dolu  : {mb(total - free):,.0f} MB  (baskalarinin isi)")
    if free < 2e9:
        raise SystemExit("\n  2 GB'tan az bos alan var — bu makinede kosmayin.")

    crit = FocalLoss(alpha=6.5, gamma=2.0)
    results = []

    print("\n" + "=" * 66)
    print(f"HIZ TESTI  (tech={args.tech_dim} fund={args.fund_dim} "
          f"seq={args.seq_len})")
    print("=" * 66)
    print(f"\n  {'batch':>6}{'tepe bellek':>14}{'ornek/sn':>12}{'durum':>10}")
    print("  " + "-" * 42)

    for bs in args.batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model = DualEncoderTransformer(
                tech_dim=args.tech_dim, fund_dim=args.fund_dim,
                seq_len=args.seq_len, d_model=64, n_heads=4, n_layers=2,
                dropout=0.15, modality='multimodal',
                fusion_type='cross_attention').to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

            xt = torch.randn(bs, args.seq_len, args.tech_dim, device=dev)
            xf = torch.randn(bs, args.seq_len, args.fund_dim, device=dev)
            y = torch.randint(0, 2, (bs, 1), device=dev).float()

            # Isinma — ilk adimlar CUDA baglami ve Adam durumu ayirir
            for _ in range(5):
                opt.zero_grad()
                crit(model(xt, xf), y).backward()
                opt.step()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.steps):
                opt.zero_grad()
                crit(model(xt, xf), y).backward()
                opt.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            sps = args.steps * bs / dt
            peak = torch.cuda.max_memory_allocated()
            print(f"  {bs:>6}{mb(peak):>11,.0f} MB{sps:>12,.0f}{'ok':>10}")
            results.append((bs, sps, peak))
            del model, opt, xt, xf, y
        except torch.cuda.OutOfMemoryError:
            print(f"  {bs:>6}{'—':>14}{'—':>12}{'SIGMADI':>10}")
            break
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  {bs:>6}{'—':>14}{'—':>12}{'SIGMADI':>10}")
                break
            raise
        finally:
            torch.cuda.empty_cache()

    if not results:
        raise SystemExit("\n  Hicbir batch sigmadi.")

    print("\n" + "=" * 66)
    print("TAHMINI EGITIM SURESI")
    print("=" * 66)
    print(f"  {args.n_train:,} dizi | {args.epochs} epoch | {args.seeds} seed")
    print(f"\n  {'batch':>6}{'epoch':>12}{'1 seed':>12}{'toplam':>12}")
    print("  " + "-" * 42)
    for bs, sps, _ in results:
        ep_min = args.n_train / sps / 60
        seed_h = ep_min * args.epochs / 60
        tot_h = seed_h * args.seeds
        print(f"  {bs:>6}{ep_min:>10.1f} dk{seed_h:>10.1f} sa{tot_h:>10.1f} sa")

    # Karar — batch 64 referans alinir (mevcut deneylerinizle ayni)
    ref = next((r for r in results if r[0] == 64), results[-1])
    bs, sps, _ = ref
    tot_h = (args.n_train / sps / 60) * args.epochs * args.seeds / 60

    print("\n" + "=" * 66)
    print("KARAR")
    print("=" * 66)
    print(f"  Bos GPU referansi: ~1.5 dk/epoch  (gecmis kosularinizdan)")
    print(f"  Bu makinede      : ~{args.n_train / sps / 60:.1f} dk/epoch")
    if tot_h < 8:
        print(f"\n  → {tot_h:.1f} saat. KOSUN, sorun yok.")
    elif tot_h < 20:
        print(f"\n  → {tot_h:.1f} saat. Gece kosar, kabul edilebilir.")
        print("    Kesilirse resume ile devam eder.")
    else:
        print(f"\n  → {tot_h:.1f} saat. COK UZUN.")
        print("    Seed sayisini dusurun ya da bos GPU bekleyin.")
    print("\n  NOT: bu olcum sentetik veriyle yapildi. Gercek kosuda veri")
    print("  yukleme de zaman alir, sure %20-40 daha uzun olabilir.")


if __name__ == '__main__':
    main()