import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, CosineAnnealingLR, LambdaLR, SequentialLR
)
from sklearn.metrics import (
    roc_auc_score, f1_score,
    precision_score, recall_score, classification_report,
    precision_recall_curve, average_precision_score,
    accuracy_score, matthews_corrcoef
)
import numpy as np
import matplotlib.pyplot as plt


def _build_warmup_cosine_scheduler(optimizer, num_warmup_epochs, num_total_epochs,
                                   min_lr_ratio=0.01):
    """
    Transformer literatüründe standart (Vaswani et al. 2017, BERT, ViT) olan
    warmup + cosine annealing scheduler.

    İlk `num_warmup_epochs` boyunca lr linear olarak 0'dan hedefe çıkar
    (Transformer'lar bu fazda kararsız oluyor — warmup yumuşatır).
    Sonra cosine ile `min_lr_ratio * lr`'ye kadar düşer.
    """
    warmup = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (epoch + 1) / max(1, num_warmup_epochs)
    )
    cosine_epochs = max(1, num_total_epochs - num_warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=min_lr_ratio * optimizer.param_groups[0]['lr']
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[num_warmup_epochs]
    )
    return scheduler


def plot_learning_curves(history, model_name):
    os.makedirs(os.path.join('visualization', 'learning_curves'), exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].set_xlabel('Epochs');
    axes[0].set_ylabel('Loss');
    axes[0].legend()

    axes[1].plot(epochs, history['train_auc'], 'b-', label='Train ROC-AUC')
    axes[1].plot(epochs, history['val_auc'], 'r-', label='Val ROC-AUC')
    axes[1].set_title(f'{model_name} - ROC-AUC')
    axes[1].set_xlabel('Epochs');
    axes[1].set_ylabel('AUC');
    axes[1].legend()

    axes[2].plot(epochs, history['train_pr_auc'], 'b-', label='Train PR-AUC')
    axes[2].plot(epochs, history['val_pr_auc'], 'r-', label='Val PR-AUC')
    axes[2].set_title(f'{model_name} - PR-AUC (imbalance-aware)')
    axes[2].set_xlabel('Epochs');
    axes[2].set_ylabel('PR-AUC');
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join('visualization', 'learning_curves', f'{model_name}_learning_curve.png'))
    plt.close()


def _unpack_batch(batch, device):
    """
    Batch'i model girdilerine dönüştürür. Dual-stream veya tek-stream destekler.
    Returns: (model_inputs_dict, labels)
    """
    labels = batch['label'].to(device).float().unsqueeze(1)
    inputs = {}

    if 'tech_seq' in batch and 'fund_seq' in batch:
        # Dual-encoder mod
        inputs['x_tech'] = batch['tech_seq'].to(device)
        inputs['x_fund'] = batch['fund_seq'].to(device)
    elif 'sequence' in batch:
        # Tek-encoder baseline
        inputs['x'] = batch['sequence'].to(device)

    return inputs, labels


def _forward_model(model, inputs):
    """Model forward — dual-stream veya tek-stream çağrılarını yönetir."""
    if 'x_tech' in inputs:
        return model(inputs['x_tech'], inputs['x_fund'])
    return model(inputs['x'])


def _evaluate_on_loader(model, loader, device, criterion=None):
    """
    Bir dataloader üzerinde modeli değerlendir, predictions ve targets döner.
    Eğer criterion verilirse loss da hesaplanır.
    """
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = _unpack_batch(batch, device)
            logits = _forward_model(model, inputs)
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                n_samples += labels.size(0)
            preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())
    avg_loss = total_loss / max(n_samples, 1) if criterion is not None else None
    return np.array(preds), np.array(targets), avg_loss


def _compute_classification_metrics(preds, targets, threshold):
    """
    Belirlenen threshold ile ACC, MCC, Precision, Recall, F1 hesaplar.
    Ranking metrikleri (ROC-AUC, PR-AUC) threshold-bağımsızdır, ayrı hesaplanır.
    """
    y_pred = (preds >= threshold).astype(int)
    y_true = targets.astype(int)
    metrics = {}
    try:
        metrics['roc_auc'] = roc_auc_score(targets, preds)
    except ValueError:
        metrics['roc_auc'] = 0.5
    try:
        metrics['pr_auc'] = average_precision_score(targets, preds)
    except ValueError:
        metrics['pr_auc'] = 0.0
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    return metrics


def find_best_threshold_mcc(y_true, y_scores, n_candidates: int = 300):
    """
    MCC'yi maksimize eden karar eşiğini bulur — SADECE validation üzerinde.

    Neden F1 yerine MCC:
      Birincil metrik MCC olarak raporlanıyor. F1'i maksimize eden eşik,
      %12 pozitif oranında recall'a doğru kayar ve modelin "her şeye pozitif de"
      dediği dejenere noktaları ödüllendirir (F1 ≈ 0.23 base-rate'te bile).
      MCC dört hücreyi birden kullandığı için bu tuzağa düşmez.

    Dejenere koruma:
      Tek sınıf tahmin eden eşikler atlanır. Bu, sinyal üretemeyen bir modelin
      (ör. tech_only) tamamen çökmüş bir eşiğe sabitlenmesini önler.

    Returns:
        (best_threshold, best_mcc)
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_scores = np.asarray(y_scores).astype(float).ravel()

    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    lo, hi = np.percentile(y_scores, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(y_scores.min()), float(y_scores.max())
    if hi <= lo:
        return 0.5, 0.0

    candidates = np.linspace(lo, hi, n_candidates)

    best_mcc, best_thresh = -2.0, 0.5
    n_valid = 0
    for t in candidates:
        y_pred = (y_scores >= t).astype(int)
        s = y_pred.sum()
        if s == 0 or s == len(y_pred):
            continue                      # dejenere eşik — atla
        n_valid += 1
        m = matthews_corrcoef(y_true, y_pred)
        if m > best_mcc:
            best_mcc, best_thresh = m, float(t)

    if n_valid == 0:                      # hiç geçerli eşik yok
        return float(np.median(y_scores)), 0.0

    return best_thresh, float(best_mcc)


def train_pytorch_model(model, train_loader, val_loader,
                        test_loader=None,
                        model_name="PyTorchModel", epochs=30,
                        device='cpu', criterion=None,
                        monitor: str = 'pr_auc',
                        lr: float = 1e-4,
                        weight_decay: float = 1e-2,
                        early_stop_patience: int = 8,
                        use_warmup: bool = False,
                        warmup_epochs: int = 3):
    """
    PyTorch eğitim döngüsü.

    monitor: 'pr_auc' veya 'roc_auc' — model seçimi ve early stopping için
             izlenecek metrik.
    lr: learning rate (default 1e-4)
    weight_decay: AdamW weight decay (default 1e-2)
    early_stop_patience: validation metric kaç epoch düşmezse dur (default 8)
    use_warmup: True ise warmup + cosine annealing scheduler kullanılır
                (Transformer literatür standardı). False ise ReduceLROnPlateau.
    warmup_epochs: warmup süresi (use_warmup=True olduğunda anlamlı, default 3)
    """
    assert monitor in ('pr_auc', 'roc_auc'), "monitor 'pr_auc' veya 'roc_auc' olmalı"
    sched_name = f'warmup({warmup_epochs})+cosine' if use_warmup else 'plateau'
    print(f"\n{model_name} Eğitimi Başlıyor ({device}) | monitor={monitor} | "
          f"lr={lr} | wd={weight_decay} | patience={early_stop_patience} | "
          f"scheduler={sched_name}...")
    model.to(device)

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_warmup:
        scheduler = _build_warmup_cosine_scheduler(
            optimizer,
            num_warmup_epochs=warmup_epochs,
            num_total_epochs=epochs,
            min_lr_ratio=0.01,
        )
        scheduler_is_plateau = False
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.5)
        scheduler_is_plateau = True

    best_metric = 0.0
    patience_counter = 0

    os.makedirs('checkpoints', exist_ok=True)
    best_model_path = os.path.join('checkpoints', f'best_{model_name.lower()}.pth')

    # ─ ÖNEMLİ: eski (farklı feature seti/mimariden kalma) checkpoint'i sil ──
    # Bu run hiçbir epoch'ta iyileşme sağlayamazsa (val metrik 0/NaN'da
    # takılırsa), aşağıdaki "if os.path.exists(best_model_path)" bloğu
    # SESSİZCE eski, farklı boyutlu bir checkpoint'i yüklemeye çalışır —
    # bu da kafa karıştırıcı bir "size mismatch" hatasıyla eğitimin
    # SONUNDA patlamasına yol açar (tüm epoch'lar boşa gitmiş olur).
    # Baştan silmek, ya temiz bir "hiç iyileşme olmadı" durumuna (checkpoint
    # yok, son epoch ağırlıkları kullanılır) ya da doğru boyutlu yeni bir
    # checkpoint'e yol açar — asla eski/yanlış boyutlu bir dosyaya değil.
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        print(f"  [Checkpoint] Eski {best_model_path} silindi (farklı feature "
              f"seti/mimariden kalma olabilirdi) — bu run temiz başlıyor.")

    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc': [], 'val_auc': [],
        'train_pr_auc': [], 'val_pr_auc': [],
    }

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            inputs, labels = _unpack_batch(batch, device)
            optimizer.zero_grad()
            logits = _forward_model(model, inputs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
            train_targets.extend(labels.cpu().numpy().flatten())

        train_loss /= len(train_loader.dataset)

        # ── Validation ────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = _unpack_batch(batch, device)
                logits = _forward_model(model, inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                val_targets.extend(labels.cpu().numpy().flatten())

        val_loss /= max(len(val_loader.dataset), 1)

        # ── Metrikler ─────────────────────────────────────────────
        try:
            train_auc = roc_auc_score(train_targets, train_preds)
            val_auc = roc_auc_score(val_targets, val_preds)
            train_pr_auc = average_precision_score(train_targets, train_preds)
            val_pr_auc = average_precision_score(val_targets, val_preds)
        except ValueError:
            train_auc = val_auc = 0.5
            train_pr_auc = val_pr_auc = 0.0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['train_pr_auc'].append(train_pr_auc)
        history['val_pr_auc'].append(val_pr_auc)

        v_bin = (np.array(val_preds) >= 0.5).astype(int)
        try:
            v_prec = precision_score(val_targets, v_bin, zero_division=0)
            v_rec = recall_score(val_targets, v_bin, zero_division=0)
        except Exception:
            v_prec = v_rec = 0.0

        print(f"Epoch {epoch + 1:02d}/{epochs} | "
              f"Loss T:{train_loss:.4f} V:{val_loss:.4f} | "
              f"ROC-AUC T:{train_auc:.3f} V:{val_auc:.3f} | "
              f"PR-AUC T:{train_pr_auc:.3f} V:{val_pr_auc:.3f} | "
              f"P:{v_prec:.3f} R:{v_rec:.3f}")

        # ── Model selection ───────────────────────────────────────
        current_metric = val_pr_auc if monitor == 'pr_auc' else val_auc
        # Scheduler: Plateau val metric ister, Cosine/SequentialLR istemez
        if scheduler_is_plateau:
            scheduler.step(current_metric)
        else:
            scheduler.step()

        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Yeni en iyi model kaydedildi ({monitor}: {current_metric:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  Early Stopping: Epoch {epoch + 1}'de duruldu "
                      f"(En iyi {monitor}: {best_metric:.4f})")
                break

    plot_learning_curves(history, model_name)

    # ═══════════════════════════════════════════════════════════════
    # Final değerlendirme: best checkpoint ile val (+ varsa test)
    # ═══════════════════════════════════════════════════════════════
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # ── Validation set üzerinde değerlendir ──
    val_preds, val_targets, _ = _evaluate_on_loader(model, val_loader, device)

    # MCC-optimal threshold: SADECE validation'dan seçilir (test'e leak yok!)
    # Neden F1 değil MCC: birincil metrik MCC olarak raporlanıyor. F1'i maksimize
    # eden eşik, sınıf dengesizliğinde recall'a doğru kayar ve dejenere
    # ("her şeye pozitif de") çözümleri ödüllendirir. Eşiği raporlanan metrikle
    # tutarlı seçmek hem metodolojik olarak doğru hem de MCC'yi iyileştirir.
    best_thresh, best_val_mcc = find_best_threshold_mcc(val_targets, val_preds)

    val_metrics = _compute_classification_metrics(val_preds, val_targets, best_thresh)

    print(f"\n{'═' * 60}")
    print(f"  {model_name} — VALIDATION SET")
    print(f"{'═' * 60}")
    print(f"  Threshold-bağımsız (ranking):")
    print(f"    ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"    PR-AUC:  {val_metrics['pr_auc']:.4f}")
    print(f"  Optimal Threshold (MCC max from VAL): {best_thresh:.4f} "
          f"(val MCC={best_val_mcc:.4f})")
    print(f"  Threshold-bağımlı (eşik={best_thresh:.4f}):")
    print(f"    Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"    MCC:       {val_metrics['mcc']:.4f}")
    print(f"    Precision: {val_metrics['precision']:.4f}")
    print(f"    Recall:    {val_metrics['recall']:.4f}")
    print(f"    F1:        {val_metrics['f1']:.4f}")

    y_pred_val = (val_preds >= best_thresh).astype(int)
    print(f"\n=== {model_name} Validation Classification Report ===")
    print(classification_report(val_targets, y_pred_val,
                                target_names=['Stable (0)', 'Risk (1)'], zero_division=0))

    # ── Test set üzerinde değerlendir (varsa) ──
    test_metrics = None
    if test_loader is not None:
        test_preds, test_targets, _ = _evaluate_on_loader(model, test_loader, device)
        # ÖNEMLİ: Test'te val'dan SEÇİLEN threshold kullanılır (test'e leak yok)
        test_metrics = _compute_classification_metrics(test_preds, test_targets, best_thresh)

        print(f"\n{'═' * 60}")
        print(f"  {model_name} — TEST SET (hold-out, 2024+)")
        print(f"{'═' * 60}")
        print(f"  Threshold-bağımsız (ranking):")
        print(f"    ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"    PR-AUC:  {test_metrics['pr_auc']:.4f}")
        print(f"  Threshold-bağımlı (val'dan seçilen eşik={best_thresh:.4f}):")
        print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"    MCC:       {test_metrics['mcc']:.4f}")
        print(f"    Precision: {test_metrics['precision']:.4f}")
        print(f"    Recall:    {test_metrics['recall']:.4f}")
        print(f"    F1:        {test_metrics['f1']:.4f}")

        y_pred_test = (test_preds >= best_thresh).astype(int)
        print(f"\n=== {model_name} Test Classification Report ===")
        print(classification_report(test_targets, y_pred_test,
                                    target_names=['Stable (0)', 'Risk (1)'], zero_division=0))

        # Val → Test generalization gap
        print(f"\n  Generalization gap (val → test):")
        print(f"    ROC-AUC: {val_metrics['roc_auc']:.4f} → {test_metrics['roc_auc']:.4f} "
              f"(Δ {test_metrics['roc_auc'] - val_metrics['roc_auc']:+.4f})")
        print(f"    PR-AUC:  {val_metrics['pr_auc']:.4f} → {test_metrics['pr_auc']:.4f} "
              f"(Δ {test_metrics['pr_auc'] - val_metrics['pr_auc']:+.4f})")
        print(f"    MCC:     {val_metrics['mcc']:.4f} → {test_metrics['mcc']:.4f} "
              f"(Δ {test_metrics['mcc'] - val_metrics['mcc']:+.4f})")

    # ── Tüm sonuçları kaydet ──
    os.makedirs('checkpoints', exist_ok=True)
    eval_path = os.path.join('checkpoints', f'{model_name.lower()}_eval.txt')
    with open(eval_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Optimal Threshold (from val): {best_thresh:.4f}\n\n")
        f.write("=== VALIDATION SET ===\n")
        for k, v in val_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        if test_metrics is not None:
            f.write("\n=== TEST SET ===\n")
            for k, v in test_metrics.items():
                f.write(f"  {k}: {v:.4f}\n")

    return {
        'model': model,
        'best_val_metric': best_metric,
        'threshold': best_thresh,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }