"""
Cross-attention Transformer için Optuna tuning.

XGBoost'la fair comparison için CA Regularized modeline 30-trial
Bayesian search uygular. Sadece 4 kritik hyperparameter aranır:
  - learning_rate
  - dropout
  - weight_decay
  - focal_alpha

d_model, n_layers, n_heads gibi mimari kararlar sabit tutulur
(çünkü onların aranması her trial'da farklı model boyutu yaratır,
karşılaştırılabilirlik bozulur ve compute patlar).
"""
import os
import torch
import warnings
import optuna
from sklearn.preprocessing import RobustScaler

from datasets.feature_engineering import (
    prepare_dataset,
    get_dual_stream_dataloaders,
    get_feature_groups,
)
from models.transformer_model import DualEncoderTransformer
from models.losses import FocalLoss
from models.pytorch_trainer import train_pytorch_model

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

optuna.logging.set_verbosity(optuna.logging.INFO)


def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def objective(trial, dataset_out, tech_cols, fund_cols):
    """
    Tek bir trial. Her trial yeni hyperparameter setiyle CA Regularized'ı eğitir,
    val PR-AUC'i geri döner. Optuna bu değeri maksimize eder.
    """
    # ── Aranan hyperparameter'lar ───────────────────────────────────
    lr           = trial.suggest_float('lr',           1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-3, 1e-1, log=True)
    dropout      = trial.suggest_float('dropout',      0.15, 0.40)
    focal_alpha  = trial.suggest_float('focal_alpha',  2.0, 8.0)

    print(f"\n  [Trial {trial.number}] lr={lr:.2e} | wd={weight_decay:.3f} | "
          f"dropout={dropout:.2f} | focal_alpha={focal_alpha:.1f}")

    # DataLoader (sabit batch_size=128, regularized config'inden)
    train_loader, val_loader, test_loader, _, _ = get_dual_stream_dataloaders(
        dataset_out, seq_len=20, batch_size=512
    )

    device = _get_device()

    # ── Sabit mimari (regularized config'iyle aynı) ────────────────
    model = DualEncoderTransformer(
        tech_dim=len(tech_cols),
        fund_dim=len(fund_cols),
        seq_len=20,
        d_model=48,
        n_heads=4,
        n_layers=1,
        ffn_dim=128,
        dropout=dropout,
        modality='multimodal',
        fusion_type='cross_attention',
    )

    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0)

    # Daha kısa eğitim — trial başına 25 epoch yeterli
    # (full pipeline 40 yapıyordu, ama trial'da hız önemli)
    try:
        result = train_pytorch_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,                     # test'e dokunmuyoruz
            model_name=f"CA_trial_{trial.number}",
            epochs=25,
            device=device,
            criterion=criterion,
            monitor='pr_auc',
            lr=lr,
            weight_decay=weight_decay,
            early_stop_patience=4,
        )
        val_pr_auc = result['val_metrics']['pr_auc']
        print(f"  [Trial {trial.number}] Val PR-AUC: {val_pr_auc:.4f}")
        return val_pr_auc
    except Exception as e:
        # Bir trial başarısız olursa devam et, low score ver
        print(f"  [Trial {trial.number}] FAILED: {e}")
        return 0.0


def run_optuna_tuning(n_trials=30):
    print("=" * 60)
    print(f"CROSS-ATTENTION OPTUNA TUNING ({n_trials} trial)")
    print("=" * 60)

    # Veri sadece bir kez yüklenir, trial'lar paylaşır
    dataset_out = prepare_dataset()
    tech_cols, fund_cols = get_feature_groups(dataset_out)

    study = optuna.create_study(
        direction='maximize',
        study_name='ca_regularized_tuning',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    study.optimize(
        lambda trial: objective(trial, dataset_out, tech_cols, fund_cols),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\n" + "=" * 60)
    print(f"  OPTUNA TUNING TAMAMLANDI")
    print("=" * 60)
    print(f"  En iyi val PR-AUC: {study.best_value:.4f}")
    print(f"  En iyi parametreler:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # En iyi config'i kaydet
    os.makedirs('checkpoints', exist_ok=True)
    with open(os.path.join('checkpoints', 'ca_tuning_best.txt'), 'w') as f:
        f.write(f"Best Val PR-AUC: {study.best_value:.4f}\n\n")
        f.write("Best Params:\n")
        for k, v in study.best_params.items():
            f.write(f"  {k}: {v}\n")

    # ── BEST CONFIG ile FINAL EVAL (test set dahil) ───────────────
    print("\n" + "=" * 60)
    print("  EN İYİ CONFIG İLE FINAL EĞİTİM + TEST EVAL")
    print("=" * 60)

    train_loader, val_loader, test_loader, _, _ = get_dual_stream_dataloaders(
        dataset_out, seq_len=20, batch_size=512
    )
    device = _get_device()

    best_model = DualEncoderTransformer(
        tech_dim=len(tech_cols),
        fund_dim=len(fund_cols),
        seq_len=20,
        d_model=48,
        n_heads=4,
        n_layers=1,
        ffn_dim=128,
        dropout=study.best_params['dropout'],
        modality='multimodal',
        fusion_type='cross_attention',
    )
    criterion = FocalLoss(alpha=study.best_params['focal_alpha'], gamma=2.0)

    final_result = train_pytorch_model(
        model=best_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,             # şimdi test'i de değerlendir
        model_name="CA_tuned_final",
        epochs=40,                            # final için daha uzun
        device=device,
        criterion=criterion,
        monitor='pr_auc',
        lr=study.best_params['lr'],
        weight_decay=study.best_params['weight_decay'],
        early_stop_patience=8,
    )

    print("\n" + "=" * 60)
    print("  FINAL SONUÇ — TUNED CROSS-ATTENTION")
    print("=" * 60)
    print(f"  Val:  PR-AUC {final_result['val_metrics']['pr_auc']:.4f} | "
          f"MCC {final_result['val_metrics']['mcc']:.4f}")
    if final_result['test_metrics']:
        print(f"  Test: PR-AUC {final_result['test_metrics']['pr_auc']:.4f} | "
              f"MCC {final_result['test_metrics']['mcc']:.4f}")

    return study, final_result


if __name__ == "__main__":
    # Standart: 30 trial. Daha hızlı denemek istersen 15-20 yap.
    study, final_result = run_optuna_tuning(n_trials=30)