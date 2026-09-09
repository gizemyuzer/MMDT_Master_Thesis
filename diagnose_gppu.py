"""
diagnose_gpu.py
────────────────
CUDA out-of-memory teşhisi: ne kadar bellek var, hangi batch boyutu sığıyor?

Tahmin yürütmek yerine ölçer. Gerçek modeli gerçek boyutlarla kurar, ileri +
geri geçiş yapar ve batch başına TEPE bellek kullanımını raporlar.

Neden gerekli: "batch'i düşür" tavsiyesi kolay ama ana faktöriyel koşunuz
batch 64 ile üretildi. Farklı batch ile koşulan bir deney onunla birebir
karşılaştırılamaz. Önce neyin sığdığını bilmek, sonra karşılaştırılabilirliği
nasıl koruyacağımıza karar vermek gerekiyor.

KULLANIM:
    python diagnose_gpu.py
    python diagnose_gpu.py --batches 16 32 64 128 --fund-dim 6
"""
import argparse
import torch

from models.transformer_model import DualEncoderTransformer
from models.losses import FocalLoss


def mb(x):
    return x / 1024 ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', type=int, nargs='+', default=[16, 32, 64, 128, 256])
    ap.add_argument('--tech-dim', type=int, default=32)
    ap.add_argument('--fund-dim', type=int, default=6)
    ap.add_argument('--seq-len', type=int, default=20)
    ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--n-heads', type=int, default=4)
    ap.add_argument('--n-layers', type=int, default=2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA yok — bu teşhis yalnızca NVIDIA GPU için anlamlı.")

    dev = torch.device('cuda')
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()

    print("═" * 70)
    print("GPU DURUMU")
    print("═" * 70)
    print(f"  Cihaz        : {props.name}")
    print(f"  Toplam bellek: {mb(props.total_memory):,.0f} MB")
    print(f"  Şu an boş    : {mb(free):,.0f} MB")
    print(f"  Kullanımda   : {mb(total - free):,.0f} MB")
    if mb(total - free) > 500:
        print("\n  ⚠️ Model kurulmadan önce bile 500 MB'tan fazla dolu.")
        print("     Başka bir süreç GPU kullanıyor olabilir (nvidia-smi ile bakın).")
    print()

    print("═" * 70)
    print("BATCH BOYUTU TARAMASI — ileri + geri geçiş")
    print("═" * 70)
    print(f"  tech_dim={args.tech_dim} fund_dim={args.fund_dim} "
          f"seq_len={args.seq_len} d_model={args.d_model}")
    print(f"\n  {'batch':>7}{'tepe bellek':>16}{'durum':>12}")
    print("  " + "-" * 37)

    largest = None
    for bs in args.batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model = DualEncoderTransformer(
                tech_dim=args.tech_dim, fund_dim=args.fund_dim,
                seq_len=args.seq_len, d_model=args.d_model,
                n_heads=args.n_heads, n_layers=args.n_layers,
                dropout=0.15, modality='multimodal',
                fusion_type='cross_attention').to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            crit = FocalLoss(alpha=6.5, gamma=2.0)

            xt = torch.randn(bs, args.seq_len, args.tech_dim, device=dev)
            xf = torch.randn(bs, args.seq_len, args.fund_dim, device=dev)
            y = torch.randint(0, 2, (bs, 1), device=dev).float()

            # İki adım: ilk adım optimizer durumunu (Adam momentleri) da ayırır,
            # tepe bellek ancak ikinci adımda gerçekçi olur.
            for _ in range(2):
                opt.zero_grad()
                loss = crit(model(xt, xf), y)
                loss.backward()
                opt.step()

            peak = torch.cuda.max_memory_allocated()
            print(f"  {bs:>7}{mb(peak):>13,.0f} MB{'sığdı':>12}")
            largest = bs
            del model, opt, xt, xf, y, loss
        except torch.cuda.OutOfMemoryError:
            print(f"  {bs:>7}{'—':>16}{'SIĞMADI':>12}")
            break
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  {bs:>7}{'—':>16}{'SIĞMADI':>12}")
                break
            raise
        finally:
            torch.cuda.empty_cache()

    print("\n" + "═" * 70)
    print("YORUM")
    print("═" * 70)
    if largest is None:
        print("  Hiçbir batch boyutu sığmadı. Bu, modelin değil ortamın sorunu:")
        print("  GPU'da başka bir süreç var ya da sürücü belleği serbest bırakmamış.")
        print("  Makineyi yeniden başlatıp tekrar deneyin.")
    elif largest >= 64:
        print(f"  Batch {largest} sığıyor. Ana faktöriyel de 64 ile koşuldu,")
        print("  yani karşılaştırılabilirlik korunabilir.")
        print("  OOM aldıysanız sorun büyük olasılıkla VERİ YÜKLEME tarafında:")
        print("  DualStreamSequenceDataset tüm dizileri RAM'de tutuyor ve bu")
        print("  deneyin eğitim seti bir yıl daha uzun. GPU değil, sistem belleği")
        print("  dolmuş olabilir — Görev Yöneticisi'nden RAM kullanımına bakın.")
    else:
        print(f"  Yalnızca batch {largest} ve altı sığıyor.")
        print("  Ana koşu 64 ile yapıldığı için doğrudan batch düşürmek")
        print("  karşılaştırmayı bozar. Gradyan biriktirme (gradient accumulation)")
        print(f"  ile mikro-batch {largest} × {64 // max(largest,1)} adım = efektif 64")
        print("  kullanmak daha doğru olur; model LayerNorm kullandığı için")
        print("  bu bölme matematiksel olarak eşdeğerdir.")


if __name__ == '__main__':
    main()