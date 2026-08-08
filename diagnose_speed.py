"""
diagnose_speed.py
─────────────────
Eğitimin NEDEN yavaş oldugunu satir satir olcer. Gercek modelini ve gercek
veriyi kullanir, tek bir egitim adiminin nerede zaman harcadigini gosterir:
  - model GPU'da mi?
  - veri GPU'ya tasiniyor mu, ne kadar suruyor?
  - forward/backward GPU'da hizli mi?
  - DataLoader batch uretimi yavas mi? (asil suphe)

KULLANIM:
    C:\\Users\\go39lop\\Desktop\\thesis_venv\\Scripts\\python.exe diagnose_speed.py
"""
import time
import torch
from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer

print("=" * 60)
print("HIZ TESHISI")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n[1] Veri yukleniyor (cache)...")
t = time.time()
df = prepare_dataset(force_refresh=False)
print(f"    {time.time()-t:.1f}s")

print("\n[2] DataLoader olusturuluyor...")
t = time.time()
train_loader, val_loader, test_loader, _, (tech_cols, fund_cols) = \
    get_dual_stream_dataloaders(df, seq_len=20, batch_size=512)
print(f"    {time.time()-t:.1f}s")

print("\n[3] Model olusturuluyor + GPU'ya tasiniyor...")
model = DualEncoderTransformer(
    tech_dim=len(tech_cols), fund_dim=len(fund_cols),
    seq_len=20, d_model=64, n_heads=4, n_layers=2,
    dropout=0.15, modality='tech_only', fusion_type='cross_attention')
t = time.time()
model.to(device)
print(f"    model.to(device): {time.time()-t:.3f}s")
# modelin gercekten GPU'da oldugunu dogrula
p = next(model.parameters())
print(f"    Model parametreleri nerede: {p.device}  <-- 'cuda:0' OLMALI")

print("\n[4] TEK BATCH testi — zaman nereye gidiyor?")
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = torch.nn.BCEWithLogitsLoss()

# ilk batch'i cek (DataLoader suresi)
t = time.time()
it = iter(train_loader)
batch = next(it)
print(f"    Ilk batch DataLoader'dan cikti: {time.time()-t:.3f}s  <-- yavassa DataLoader sorunu")

# batch'i GPU'ya tasi
t = time.time()
xt = batch['tech_seq'].to(device)
xf = batch['fund_seq'].to(device)
y = batch['label'].to(device).float().unsqueeze(1)
torch.cuda.synchronize() if device.type == 'cuda' else None
print(f"    Batch GPU'ya tasindi: {time.time()-t:.3f}s")

# forward + backward
t = time.time()
opt.zero_grad()
out = model(x_tech=xt, x_fund=xf)
loss = crit(out, y)
loss.backward()
opt.step()
torch.cuda.synchronize() if device.type == 'cuda' else None
print(f"    Forward+backward (GPU): {time.time()-t:.3f}s  <-- 0.5s'ten AZ olmali")

print("\n[5] 10 BATCH hiz testi (DataLoader dahil)...")
t = time.time()
n = 0
for batch in train_loader:
    xt = batch['tech_seq'].to(device)
    xf = batch['fund_seq'].to(device)
    y = batch['label'].to(device).float().unsqueeze(1)
    opt.zero_grad()
    out = model(x_tech=xt, x_fund=xf)
    loss = crit(out, y)
    loss.backward()
    opt.step()
    n += 1
    if n >= 10:
        break
torch.cuda.synchronize() if device.type == 'cuda' else None
dt = time.time() - t
print(f"    10 batch: {dt:.2f}s = {dt/10:.3f}s/batch")

# toplam batch sayisi ve tahmini epoch suresi
total_batches = len(train_loader)
print(f"\n[6] Tahmini epoch suresi:")
print(f"    Toplam batch (batch=512): {total_batches}")
print(f"    {dt/10:.3f}s/batch x {total_batches} = {(dt/10)*total_batches/60:.1f} dakika/epoch")

print("\n" + "=" * 60)
print("YORUM:")
print("  - [4] Forward+backward > 2s ise: model GPU'da degil ya da cok buyuk")
print("  - [4] Ilk batch DataLoader > 5s ise: veri uretimi yavas")
print("  - [5] s/batch cok yuksekse (>0.5): darbogaz DataLoader/CPU'da")
print("  - Model parametreleri 'cuda:0' degilse: model CPU'da kalmis")
print("=" * 60)