import os
import torch
import warnings
from sklearn.preprocessing import RobustScaler

# Dataset and Feature Engineering
from datasets.feature_engineering import (
    prepare_dataset,
    get_sequence_dataloaders,  # eski tek-stream loader (baseline için)
    get_dual_stream_dataloaders,  # YENİ: dual-stream loader
    get_feature_groups,  # tech/fund split helper
    ABSOLUTE_COLS,
)
from datasets.eda import run_eda

# Models and Training
from models.xgboost_model import train_xgboost
from models.transformer_model import TimeSeriesTransformer, DualEncoderTransformer, DualStreamRiskModel
# NOT: LSTM tezden çıkarıldı — yerine gated cross-attention eklendi.
# Geri almak istersen: from models.lstm_model import DualEncoderLSTM
from models.losses import FocalLoss, DrawdownFocalLoss
from models.pytorch_trainer import train_pytorch_model, find_best_threshold_mcc

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '8'


def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _compute_focal_alpha(train_loader):
    """Train set'teki sınıf dengesizliğine göre FocalLoss alpha hesaplar."""
    all_labels = torch.cat([b['label'] for b in train_loader])
    n_neg = (all_labels == 0).sum().float()
    n_pos = (all_labels == 1).sum().float()
    return (n_neg / torch.clamp(n_pos, min=1.0)).item()


def run_xgboost_pipeline(dataset_out):
    """
    XGBoost baseline — Transformer'larla aynı feature set'i flat olarak alır.
    XGBoost'un dual-encoder gibi modalite ayırma yapısı yok; bu yüzden
    tech_cols + fund_cols hepsi tek vector'da. Karşılaştırmanın amacı:
    "modalite ayrımı + cross-attention, flat boosting'e ne kazandırıyor?"

    Hold-out test seti üzerinde de değerlendirir (val'dan seçilen threshold ile).
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        accuracy_score, matthews_corrcoef,
        precision_score, recall_score, f1_score,
        precision_recall_curve, classification_report
    )
    import numpy as np

    # Transformer pipeline'larıyla aynı feature kümesi (adil karşılaştırma için)
    tech_cols, fund_cols = get_feature_groups(dataset_out)
    feature_cols = tech_cols + fund_cols
    print(f"\n  XGBoost feature set → tech: {len(tech_cols)} | "
          f"fund: {len(fund_cols)} | toplam: {len(feature_cols)}")

    train_mask = (dataset_out.index >= '2010-01-01') & (dataset_out.index <= '2019-12-31')
    val_mask = (dataset_out.index >= '2020-01-01') & (dataset_out.index <= '2021-12-31')
    test_mask = (dataset_out.index >= '2022-01-01') & (dataset_out.index <= '2024-12-31')

    train_subset = dataset_out[train_mask]
    val_subset = dataset_out[val_mask]
    test_subset = dataset_out[test_mask]

    print(f"  Train: {len(train_subset):,} | Val: {len(val_subset):,} | Test: {len(test_subset):,}")

    train_medians = train_subset[feature_cols].median()
    X_train_raw = train_subset[feature_cols].fillna(train_medians).values
    y_train_raw = train_subset['Target'].values
    X_val_raw = val_subset[feature_cols].fillna(train_medians).values
    y_val_raw = val_subset['Target'].values
    X_test_raw = test_subset[feature_cols].fillna(train_medians).values
    y_test_raw = test_subset['Target'].values

    # Tarih bilgisini panel-aware CV için XGBoost'a geçir
    train_dates = train_subset.index

    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_val_sc = scaler.transform(X_val_raw)
    X_test_sc = scaler.transform(X_test_raw)

    # use_cached_params=False: evren (110 tech → 400 multi-sector), split
    # (2015-2021 → 2010-2019) ve feature seti değişti. Eski Optuna
    # parametreleri artık geçerli değil — yeniden aranmalı, aksi halde
    # "fair comparison" iddiası çöker.
    xgb_model, best_params, (val_roc_auc, val_pr_auc) = train_xgboost(
        X_train_sc, y_train_raw,
        X_val_sc, y_val_raw,
        feature_names=feature_cols,
        train_dates=train_dates,
        use_cached_params=False,
    )

    # ── Test set değerlendirmesi (val'dan seçilen threshold ile) ──
    val_probs = xgb_model.predict_proba(X_val_sc)[:, 1]
    test_probs = xgb_model.predict_proba(X_test_sc)[:, 1]

    # MCC-optimal threshold val'dan seçilir (test'e leak yok)
    # Transformer'lar da aynı kriteri kullanıyor → fair comparison
    best_thresh, _ = find_best_threshold_mcc(y_val_raw, val_probs)

    # Val metrikleri
    y_val_pred = (val_probs >= best_thresh).astype(int)
    val_metrics = {
        'roc_auc': val_roc_auc,
        'pr_auc': val_pr_auc,
        'accuracy': accuracy_score(y_val_raw, y_val_pred),
        'mcc': matthews_corrcoef(y_val_raw, y_val_pred),
        'precision': precision_score(y_val_raw, y_val_pred, zero_division=0),
        'recall': recall_score(y_val_raw, y_val_pred, zero_division=0),
        'f1': f1_score(y_val_raw, y_val_pred, zero_division=0),
    }

    # Test metrikleri (val'dan seçilen threshold ile)
    y_test_pred = (test_probs >= best_thresh).astype(int)
    test_metrics = {
        'roc_auc': roc_auc_score(y_test_raw, test_probs),
        'pr_auc': average_precision_score(y_test_raw, test_probs),
        'accuracy': accuracy_score(y_test_raw, y_test_pred),
        'mcc': matthews_corrcoef(y_test_raw, y_test_pred),
        'precision': precision_score(y_test_raw, y_test_pred, zero_division=0),
        'recall': recall_score(y_test_raw, y_test_pred, zero_division=0),
        'f1': f1_score(y_test_raw, y_test_pred, zero_division=0),
    }

    print(f"\n{'═' * 60}")
    print(f"  XGBoost — VALIDATION SET")
    print(f"{'═' * 60}")
    print(f"  ROC-AUC: {val_metrics['roc_auc']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f}")
    print(f"  Threshold: {best_thresh:.4f}")
    print(f"  ACC: {val_metrics['accuracy']:.4f} | MCC: {val_metrics['mcc']:.4f} | "
          f"F1: {val_metrics['f1']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f}")

    print(f"\n{'═' * 60}")
    print(f"  XGBoost — TEST SET (hold-out, 2024+)")
    print(f"{'═' * 60}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"  ACC: {test_metrics['accuracy']:.4f} | MCC: {test_metrics['mcc']:.4f} | "
          f"F1: {test_metrics['f1']:.4f} | P: {test_metrics['precision']:.4f} | R: {test_metrics['recall']:.4f}")
    print(f"\n=== XGBoost Test Classification Report ===")
    print(classification_report(y_test_raw, y_test_pred,
                                target_names=['Stable (0)', 'Risk (1)'], zero_division=0))

    print(f"\n  Generalization gap (val → test):")
    print(f"    ROC-AUC: {val_metrics['roc_auc']:.4f} → {test_metrics['roc_auc']:.4f} "
          f"(Δ {test_metrics['roc_auc'] - val_metrics['roc_auc']:+.4f})")
    print(f"    PR-AUC:  {val_metrics['pr_auc']:.4f} → {test_metrics['pr_auc']:.4f} "
          f"(Δ {test_metrics['pr_auc'] - val_metrics['pr_auc']:+.4f})")
    print(f"    MCC:     {val_metrics['mcc']:.4f} → {test_metrics['mcc']:.4f} "
          f"(Δ {test_metrics['mcc'] - val_metrics['mcc']:+.4f})")

    return {
        'model': xgb_model,
        'threshold': best_thresh,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }


def run_single_encoder_pipeline(dataset_out):
    """Baseline: tek-encoder Transformer (ablation amaçlı)."""
    print("\n" + "=" * 60)
    print("PHASE 4a: BASELINE — TEK-ENCODER TRANSFORMER")
    print("=" * 60)

    train_loader, val_loader, test_loader, _ = get_sequence_dataloaders(
        dataset_out, seq_len=20, batch_size=512
    )

    feature_cols = [c for c in dataset_out.columns
                    if c not in ['Target', 'Ticker', 'Sector'] + ABSOLUTE_COLS]
    device = _get_device()

    model = TimeSeriesTransformer(
        feature_dim=len(feature_cols),
        seq_len=20,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )

    alpha_weight = _compute_focal_alpha(train_loader)
    criterion = FocalLoss(alpha=alpha_weight, gamma=2.0)

    print(f"  Feature dim: {len(feature_cols)} | seq_len: 20 | d_model: 64")
    print(f"  FocalLoss alpha: {alpha_weight:.2f} | Device: {device}")

    return train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="SingleEncoderBaseline",
        epochs=30,
        device=device,
        criterion=criterion,
        monitor='pr_auc',
    )


def run_modality_ablation(dataset_out, modality: str):
    """
    Modality ablation pipeline'ı.

    Tezin "multi-modal yaklaşımın katma değeri" iddiasını test eden ablation.
    Üç modu destekler:
      - 'tech_only':   sadece price-based features (technical encoder)
      - 'fund_only':   sadece fundamental + macro features (fundamental encoder)
      - 'multimodal':  her ikisi + cross-attention fusion (tezin ana modeli)

    Argüman zinciri:
      if multimodal PR-AUC > max(tech_only, fund_only):
          → multi-modal yaklaşımın katma değeri kanıtlanmış
      else:
          → modaliteler arası etkileşim modellenemiyor (yine geçerli bir bulgu)
    """
    assert modality in ('tech_only', 'fund_only', 'multimodal'), \
        f"modality 'tech_only', 'fund_only' veya 'multimodal' olmalı"

    print("\n" + "=" * 60)
    print(f"MODALITY ABLATION: {modality.upper()}")
    print("=" * 60)

    train_loader, val_loader, test_loader, scalers, (tech_cols, fund_cols) = get_dual_stream_dataloaders(
        dataset_out, seq_len=20, batch_size=512
    )

    device = _get_device()

    # Aynı sınıf, modality'ye göre içsel yapı değişiyor
    model = DualEncoderTransformer(
        tech_dim=len(tech_cols),
        fund_dim=len(fund_cols),
        seq_len=20,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.15,
        modality=modality,
        fusion_type='cross_attention',  # multimodal modunda kullanılır
    )

    alpha_weight = _compute_focal_alpha(train_loader)
    criterion = FocalLoss(alpha=alpha_weight, gamma=2.0)

    n_params = sum(p.numel() for p in model.parameters())
    if modality == 'tech_only':
        print(f"  Mod: TECH-ONLY (sadece price-based features)")
        print(f"  tech_dim: {len(tech_cols)} | fund_dim: kullanılmıyor")
    elif modality == 'fund_only':
        print(f"  Mod: FUND-ONLY (sadece fundamental + macro features)")
        print(f"  tech_dim: kullanılmıyor | fund_dim: {len(fund_cols)}")
    else:
        print(f"  Mod: MULTIMODAL (her ikisi + cross-attention)")
        print(f"  tech_dim: {len(tech_cols)} | fund_dim: {len(fund_cols)}")
    print(f"  Toplam parametre: {n_params:,}")
    print(f"  FocalLoss alpha: {alpha_weight:.2f} | Device: {device}")

    model_name = f"Modality_{modality}"
    return train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name=model_name,
        epochs=30,
        device=device,
        criterion=criterion,
        monitor='pr_auc',
    )


def run_dual_encoder_pipeline(dataset_out, fusion_type: str = 'cross_attention'):
    """
    Tezdeki ana mimari: iki ayrı encoder + cross-attention fusion.
    fusion_type: 'cross_attention' (ana model) veya 'concat' (ablation).
    """
    print("\n" + "=" * 60)
    print(f"PHASE 4b: DUAL-ENCODER TRANSFORMER (fusion={fusion_type})")
    print("=" * 60)

    train_loader, val_loader, test_loader, scalers, (tech_cols, fund_cols) = get_dual_stream_dataloaders(
        dataset_out, seq_len=20, batch_size=512
    )

    device = _get_device()

    model = DualEncoderTransformer(
        tech_dim=len(tech_cols),
        fund_dim=len(fund_cols),
        seq_len=20,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.15,        # ← merdivenle aynı (açıkça yazıldı: default'a bağımlı kalmasın)
        fusion_type=fusion_type,
    )

    alpha_weight = _compute_focal_alpha(train_loader)
    criterion = FocalLoss(alpha=alpha_weight, gamma=2.0)

    print(f"  tech_dim: {len(tech_cols)} | fund_dim: {len(fund_cols)} | "
          f"seq_len: 20 | d_model: 64 | fusion: {fusion_type}")
    print(f"  FocalLoss alpha: {alpha_weight:.2f} | Device: {device}")

    model_name = f"DualEncoder_{fusion_type}"
    return train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name=model_name,
        epochs=30,
        device=device,
        criterion=criterion,
        monitor='pr_auc',
    )


def run_gated_cross_attention(dataset_out):
    """
    Gated cross-attention — füzyon merdiveninin üçüncü basamağı.

    Tezdeki rolü:
        concat (statik)  →  cross_attention (dinamik)  →  gated (bağlam-duyarlı)

    ⚠️ KONFİGÜRASYON UYUMU — KRİTİK:
    Bu pipeline, merdivenin diğer iki basamağıyla BİREBİR aynı ayarları kullanır:
        concat          : run_dual_encoder_pipeline   → d_model=64, n_layers=2, dropout=0.15
        cross_attention : run_modality_ablation       → d_model=64, n_layers=2, dropout=0.15
        gated (burası)  :                             → d_model=64, n_layers=2, dropout=0.15
    Böylece üç sonuç arasındaki fark SADECE füzyon mekanizmasından gelir.
    Kapasite veya regularizasyon farkı karıştırıcı değişken olmaz.

    (Regularize edilmiş config'teki gated karşılaştırması için
     run_gated_with_warmup'a bak — o da CA_regularized_warmup ile eşleşir.)

    Literatür hizası: Zong & Zhou (2024), MSGCA — tezin en yakın mimari öncülü.
    Onlar fiyat + haber + graf üçlüsüne uyguluyor; burada fiyat + fundamental
    çiftine uyarlanıyor.
    """
    print("\n" + "=" * 60)
    print("PHASE 4d: GATED CROSS-ATTENTION (merdiven config)")
    print("    concat → cross-attention → gated  (üçü de d_model=64, n_layers=2)")
    print("=" * 60)

    train_loader, val_loader, test_loader, scalers, (tech_cols, fund_cols) = get_dual_stream_dataloaders(
        dataset_out, seq_len=20, batch_size=512
    )

    device = _get_device()

    model = DualEncoderTransformer(
        tech_dim=len(tech_cols),
        fund_dim=len(fund_cols),
        seq_len=20,
        d_model=64,          # ← merdivenle aynı
        n_heads=4,
        n_layers=2,          # ← merdivenle aynı
        dropout=0.15,        # ← merdivenle aynı
        modality='multimodal',
        fusion_type='gated_cross_attention',
    )

    alpha_weight = _compute_focal_alpha(train_loader)
    criterion = FocalLoss(alpha=alpha_weight, gamma=2.0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  tech_dim: {len(tech_cols)} | fund_dim: {len(fund_cols)} | "
          f"seq_len: 20 | d_model: 64 | n_layers: 2 | dropout: 0.15")
    print(f"  fusion: gated_cross_attention (öğrenilebilir kapı)")
    print(f"  Toplam parametre: {n_params:,}")
    print(f"  FocalLoss alpha: {alpha_weight:.2f} | Device: {device}")

    result = train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="DualEncoder_gated_cross_attention",
        epochs=30,           # ← merdivenle aynı (modality ablation da 30)
        device=device,
        criterion=criterion,
        monitor='pr_auc',
    )

    # Kapı istatistiği — tez için yorumlanabilirlik verisi
    if hasattr(model, 'last_gate_t'):
        print(f"\n  [Kapı analizi] son batch ortalama kapı değerleri:")
        print(f"    tech ← fund yönü: {model.last_gate_t:.4f}")
        print(f"    fund ← tech yönü: {model.last_gate_f:.4f}")
        print(f"    (0'a yakın: karşı modalite bastırılıyor | "
              f"1'e yakın: tam karışım)")

    return result


def run_dual_encoder_regularized(dataset_out, fusion_type: str = 'cross_attention'):
    """
    Cross-attention RERUN — overfitting'e karşı agresif regularization.

    İlk run'da train PR-AUC 0.85 / val PR-AUC 0.18 — model train'i ezberliyordu.
    Bu versiyon şu değişikliklerle aynı mimariyi tekrar deniyor:
      - dropout: 0.15 → 0.30
      - d_model: 64 → 48 (kapasite azaltma)
      - n_layers: 2 → 1
      - weight_decay: 1e-2 → 5e-2
      - early_stopping patience: 8 → 4
      - learning_rate: 1e-4 → 3e-5
    """
    print("\n" + "=" * 60)
    print(f"PHASE 4c: REGULARIZED DUAL-ENCODER (fusion={fusion_type})")
    print("    overfitting fix: smaller model + higher dropout + slower lr")
    print("=" * 60)

    train_loader, val_loader, test_loader, scalers, (tech_cols, fund_cols) = get_dual_stream_dataloaders(
        dataset_out, seq_len=20, batch_size=512
    )

    device = _get_device()

    # ── Reduced capacity model ──────────────────────────────────────
    model = DualEncoderTransformer(
        tech_dim=len(tech_cols),
        fund_dim=len(fund_cols),
        seq_len=20,
        d_model=48,  # 64 → 48
        n_heads=4,
        n_layers=1,  # 2 → 1
        ffn_dim=128,  # 256 → 128
        dropout=0.30,  # 0.15 → 0.30
        fusion_type=fusion_type,
    )

    alpha_weight = _compute_focal_alpha(train_loader)
    criterion = FocalLoss(alpha=alpha_weight, gamma=2.0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  tech_dim: {len(tech_cols)} | fund_dim: {len(fund_cols)} | "
          f"seq_len: 20 | d_model: 48 | n_layers: 1 | dropout: 0.30")
    print(f"  Toplam parametre: {n_params:,} (önceki ~3x daha büyüktü)")
    print(f"  FocalLoss alpha: {alpha_weight:.2f} | Device: {device}")

    model_name = f"DualEncoder_{fusion_type}_regularized"
    return train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name=model_name,
        epochs=40,  # daha uzun, küçük model + düşük lr
        device=device,
        criterion=criterion,
        monitor='pr_auc',
        lr=3e-5,  # 1e-4 → 3e-5 (daha yavaş öğrenme)
        weight_decay=5e-2,  # 1e-2 → 5e-2 (daha güçlü L2)
        early_stop_patience=4,  # 8 → 4 (overfitting'i erken kes)
    )


# ═══════════════════════════════════════════════════════════════════════
# WARMUP PIPELINE'LARI
# Gemini feedback'ine yanıt: Transformer literatür standardı olan
# warmup + cosine annealing scheduler. Transformer'lar ilk epoch'larda
# kararsız (Vaswani et al. 2017'den beri bilinen).
# ═══════════════════════════════════════════════════════════════════════

def run_ca_tuned_with_warmup(dataset_out):
    """
    Cross-Attention TUNED + warmup + cosine annealing.

    Optuna'nın bulduğu en iyi 4 hyperparameter (lr, wd, dropout, focal_alpha)
    ile, üstüne warmup eklenmiş hali. Bu, "fair comparison + best practice"
    kombinasyonu.

    Beklenen: val→test gap önemli ölçüde küçülür, test PR-AUC ve MCC artar.
    """
    print("\n" + "=" * 60)
    print("CA TUNED + WARMUP (literatur best practice)")
    print("  Optuna best params + linear warmup + cosine annealing")
    print("=" * 60)

    train_loader, val_loader, test_loader, _, (tech_cols, fund_cols) = \
        get_dual_stream_dataloaders(dataset_out, seq_len=20, batch_size=512)

    device = _get_device()

    # ── Optuna'nın bulduğu en iyi config ──────────────────────────
    # ⚠️ UYARI: Aşağıdaki lr/wd/dropout değerleri ESKİ evrenden (110 tech
    # hissesi, 2015-2021 split) yapılan Optuna aramasından geliyor. Yeni evren
    # (400 hisse, 9 sektör, 2010-2019 train) için yeniden aranmadılar.
    # Tez metninde bu pipeline'ı "tuned" diye sunma — ya yeniden Optuna koştur
    # ya da "eski evrenden aktarılan config" olarak etiketle.
    model = DualEncoderTransformer(
        tech_dim=len(tech_cols),
        fund_dim=len(fund_cols),
        seq_len=20,
        d_model=48,
        n_heads=4,
        n_layers=1,
        ffn_dim=128,
        dropout=0.276,  # Optuna seçimi
        modality='multimodal',
        fusion_type='cross_attention',
    )

    # Focal alpha ARTIK veriden hesaplanıyor (diğer 7 pipeline ile tutarlı).
    # Eskiden alpha=6.143 hardcoded'du — o değer 110-hisse evreninin sınıf
    # dengesinden geliyordu, yeni evrende (%12.58 pozitif) geçerli değil.
    alpha_weight = _compute_focal_alpha(train_loader)
    criterion = FocalLoss(alpha=alpha_weight, gamma=2.0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ESKİ evren Optuna config: lr=1.275e-04 | wd=0.0391 | dropout=0.276")
    print(f"  ⚠️ Bu parametreler 110-hisse evreninden — yeni evrende yeniden aranmadı")
    print(f"  FocalLoss alpha: {alpha_weight:.2f} (veriden hesaplandı)")
    print(f"  Yeni eklenti: warmup_epochs=3 + cosine annealing")
    print(f"  Toplam parametre: {n_params:,} | Device: {device}")

    return train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="CA_tuned_warmup",
        epochs=40,
        device=device,
        criterion=criterion,
        monitor='pr_auc',
        lr=1.275e-04,  # Optuna seçimi
        weight_decay=0.0391,  # Optuna seçimi
        early_stop_patience=8,
        use_warmup=True,  # YENİ
        warmup_epochs=3,
    )


def run_ca_regularized_with_warmup(dataset_out):
    """
    CA Regularized + warmup. Manuel config'in warmup'lı versiyonu.

    Bu, "Optuna olmadan da warmup ne kadar yardım eder?" sorusunu cevaplar.
    Tezdeki "warmup'ın katma değeri" ablation'u için gerekli.
    """
    print("\n" + "=" * 60)
    print("CA REGULARIZED + WARMUP")
    print("  Manual config + linear warmup + cosine annealing")
    print("=" * 60)

    train_loader, val_loader, test_loader, _, (tech_cols, fund_cols) = \
        get_dual_stream_dataloaders(dataset_out, seq_len=20, batch_size=512)

    device = _get_device()

    model = DualEncoderTransformer(
        tech_dim=len(tech_cols),
        fund_dim=len(fund_cols),
        seq_len=20,
        d_model=48,
        n_heads=4,
        n_layers=1,
        ffn_dim=128,
        dropout=0.30,
        modality='multimodal',
        fusion_type='cross_attention',
    )

    alpha_weight = _compute_focal_alpha(train_loader)
    criterion = FocalLoss(alpha=alpha_weight, gamma=2.0)

    print(f"  Manual config: lr=3e-5 | wd=5e-2 | dropout=0.30")
    print(f"  Yeni eklenti: warmup_epochs=3 + cosine annealing")
    print(f"  Device: {device}")

    return train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="CA_regularized_warmup",
        epochs=40,
        device=device,
        criterion=criterion,
        monitor='pr_auc',
        lr=3e-5,
        weight_decay=5e-2,
        early_stop_patience=4,
        use_warmup=True,
        warmup_epochs=3,
    )


def run_gated_with_warmup(dataset_out):
    """
    Gated cross-attention + warmup + cosine annealing.

    run_ca_regularized_with_warmup ile BİREBİR aynı ayarlar; tek fark füzyon
    mekanizması (cross_attention → gated_cross_attention). Bu ikisinin
    karşılaştırması, kapı mekanizmasının katkısını izole eder.
    """
    print("\n" + "=" * 60)
    print("GATED CROSS-ATTENTION + WARMUP")
    print("    ayarlar CA_regularized_warmup ile aynı — tek fark: kapı")
    print("=" * 60)

    train_loader, val_loader, test_loader, _, (tech_cols, fund_cols) = \
        get_dual_stream_dataloaders(dataset_out, seq_len=20, batch_size=512)

    device = _get_device()

    model = DualEncoderTransformer(
        tech_dim=len(tech_cols),
        fund_dim=len(fund_cols),
        seq_len=20,
        d_model=48,
        n_heads=4,
        n_layers=1,
        ffn_dim=128,
        dropout=0.30,
        modality='multimodal',
        fusion_type='gated_cross_attention',
    )

    alpha_weight = _compute_focal_alpha(train_loader)
    criterion = FocalLoss(alpha=alpha_weight, gamma=2.0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  d_model=48 | n_layers=1 | dropout=0.30 | fusion=gated_cross_attention")
    print(f"  lr=3e-5 | wd=5e-2 | warmup_epochs=3 + cosine annealing")
    print(f"  Toplam parametre: {n_params:,} | Device: {device}")

    result = train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="Gated_cross_attention_warmup",
        epochs=40,
        device=device,
        criterion=criterion,
        monitor='pr_auc',
        lr=3e-5,
        weight_decay=5e-2,
        early_stop_patience=4,
        use_warmup=True,
        warmup_epochs=3,
    )

    if hasattr(model, 'last_gate_t'):
        print(f"\n  [Kapı analizi] tech←fund: {model.last_gate_t:.4f} | "
              f"fund←tech: {model.last_gate_f:.4f}")

    return result


def run_mlp_conditioned_pipeline(dataset_out):
    """
    PHASE 7: XGBoost-Killer Dual-Stream (MLP Conditioned) Model + DrawdownFocalLoss
    """
    print("\n" + "=" * 60)
    print("PHASE 7: DUAL-STREAM MLP CONDITIONED RISK MODEL")
    print("  Statik Fundamental Profil + Teknik Zaman Serisi + DrawdownFocalLoss")
    print("=" * 60)

    train_loader, val_loader, test_loader, scalers, (tech_cols, fund_cols) = get_dual_stream_dataloaders(
        dataset_out, seq_len=20, batch_size=512
    )

    device = _get_device()

    model = DualStreamRiskModel(
        tech_input_dim=len(tech_cols),
        fund_input_dim=len(fund_cols),
        seq_len=20,
        hidden_dim=64,
        num_heads=4,
        dropout=0.15
    )

    alpha_weight = _compute_focal_alpha(train_loader)
    criterion = DrawdownFocalLoss(alpha=alpha_weight, gamma=2.0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  tech_dim: {len(tech_cols)} | fund_dim: {len(fund_cols)} | seq_len: 20 | hidden_dim: 64")
    print(f"  fusion: Gating + Cross-Attention Conditioned")
    print(f"  Toplam parametre: {n_params:,}")
    print(f"  DrawdownFocalLoss alpha: {alpha_weight:.2f} | Device: {device}")

    return train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="DualStream_MLP_Conditioned",
        epochs=40,
        device=device,
        criterion=criterion,
        monitor='pr_auc',
        lr=5e-5,  # Biraz daha yavaş öğrenme
        weight_decay=1e-2,
        early_stop_patience=6,
        use_warmup=True,
        warmup_epochs=3,
    )


if __name__ == "__main__":
    # 1. Veri Hazırlama
    dataset_out = prepare_dataset(force_refresh=False)

    # 2. EDA
    run_eda(dataset_out)

    # ═══════════════════════════════════════════════════════════════
    # 3. BASELINE: XGBoost (gradient boosting)
    # ═══════════════════════════════════════════════════════════════
    run_xgboost_pipeline(dataset_out)

    # ═══════════════════════════════════════════════════════════════
    # 4. MODALITY ABLATION
    # Single-modal vs multi-modal
    # ═══════════════════════════════════════════════════════════════

    # 4a. Sadece price-based features
    run_modality_ablation(dataset_out, modality='tech_only')

    # 4b. Sadece fundamental + macro features
    run_modality_ablation(dataset_out, modality='fund_only')

    # 4c. Her ikisi + cross-attention fusion (TEZ ANA MODELİ)
    run_modality_ablation(dataset_out, modality='multimodal')

    # ═══════════════════════════════════════════════════════════════
    # 5. EK ABLATION'LAR (mimari karşılaştırmaları)
    # ═══════════════════════════════════════════════════════════════

    # 5a. Tek-encoder Transformer (concat-only baseline, sequence learning testi)
    run_single_encoder_pipeline(dataset_out)

    # 5b. Multimodal + concat fusion (cross-attention'sız ablation)
    run_dual_encoder_pipeline(dataset_out, fusion_type='concat')

    # 5c. Gated cross-attention (füzyon merdiveninin 3. basamağı)
    #     concat → cross-attention → gated  |  Zong & Zhou (2024) MSGCA hizalı
    run_gated_cross_attention(dataset_out)

    # 5d. Cross-attention regularized (overfitting fix denemesi)
    run_dual_encoder_regularized(dataset_out, fusion_type='cross_attention')

    # ═══════════════════════════════════════════════════════════════
    # 6. WARMUP EKLEMELERİ (literatur best practice — Gemini feedback)
    # Transformer kararsızlığını yumuşatır, val→test gap'i küçültür
    # ═══════════════════════════════════════════════════════════════

    # 6a. CA Tuned + warmup (en iyi config + best practice)
    run_ca_tuned_with_warmup(dataset_out)

    # 6b. CA Regularized + warmup (manuel config karşılaştırma)
    run_ca_regularized_with_warmup(dataset_out)

    # 6c. Gated cross-attention + warmup
    #     CA_regularized_warmup ile birebir aynı ayar — tek fark kapı mekanizması
    run_gated_with_warmup(dataset_out)

    # ═══════════════════════════════════════════════════════════════
    # 7. XGBOOST KILLER MİMARİSİ (MLP Conditioned + DrawdownFocalLoss)
    # (664Gizem745)═══════════════════════════════════════════════════════════════
    run_mlp_conditioned_pipeline(dataset_out)


    print("\n" + "=" * 60)
    print("=> Tüm Pipeline Tamamlandı! Grafikler visualization/ klasöründe.")
    print("=" * 60)