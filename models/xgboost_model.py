import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score


def _make_panel_walk_forward_folds(train_dates, n_splits=5):
    """
    Panel data için tarih bazlı walk-forward CV fold'ları üretir.

    Standart TimeSeriesSplit fold'ları satır indeksine göre böler — ama
    bizim verimiz panel data (130 ticker × günlük satırlar yığılmış).
    Satır bazlı bölme aynı tarihteki farklı ticker'ları farklı fold'lara
    dağıtarak optimistik bias yaratır.

    Bu fonksiyon tarihe göre böler: bir tarih cut-off'unun öncesi train,
    sonrası val. Aynı tarihin tüm satırları (tüm ticker'lar) aynı fold'da.

    Args:
        train_dates: train set'in tarih indeksi (panel: tekrarlı tarihler)
        n_splits: kaç fold

    Yields:
        (train_idx, val_idx) tuple'ları — orijinal X_train array'inde pozisyon
    """
    train_dates = pd.to_datetime(train_dates)
    unique_dates = np.sort(train_dates.unique())
    n = len(unique_dates)

    # Expanding window: her fold önceki tüm tarihleri train olarak kullanır,
    # sonraki bir dilimi val olarak. Klasik walk-forward.
    fold_size = n // (n_splits + 1)

    for i in range(n_splits):
        train_end_idx = fold_size * (i + 1)
        val_end_idx = fold_size * (i + 2) if i < n_splits - 1 else n

        train_cutoff = unique_dates[train_end_idx]
        val_cutoff = unique_dates[val_end_idx - 1]

        train_mask = train_dates < train_cutoff
        val_mask = (train_dates >= train_cutoff) & (train_dates <= val_cutoff)

        train_pos = np.where(train_mask)[0]
        val_pos = np.where(val_mask)[0]

        if len(train_pos) > 0 and len(val_pos) > 0:
            yield train_pos, val_pos


# ═════════════════════════════════════════════════════════════════════
# CACHED OPTUNA RESULTS
# Optuna arama sonucu bulunan en iyi parametreler. Her seferinde yeniden
# aramak yerine bunları kullan → ~30 dk yerine ~30 saniyede biter.
# Feature set veya universe değişirse `use_cached_params=False` ile
# tekrar aratılmalı.
#
# Bulunma tarihi: Run 3 (2026-06, 100 trials, panel-aware walk-forward CV)
# En iyi CV PR-AUC: 0.1751
# ═════════════════════════════════════════════════════════════════════
CACHED_BEST_PARAMS = {
    'max_depth': 2,
    'learning_rate': 0.07109623758638484,
    'n_estimators': 1400,
    'subsample': 0.5270283576589949,
    'colsample_bytree': 0.5522281956441051,
    'gamma': 0.40233427463805566,
    'min_child_weight': 5,
    'reg_alpha': 0.08623037266720135,
    'reg_lambda': 0.00016818654615909803,
    'scale_pos_weight': 1.8448869663537082,
    'tree_method': 'hist',
    'device':'cuda'
}
CACHED_BEST_CV_PRAUC = 0.1863


def _run_optuna_search(X_train, y_train, train_dates, base_spw, n_trials=50):
    """
    Optuna Bayesian search çalıştırır. use_cached_params=False verildiğinde
    train_xgboost tarafından çağrılır.

    Feature set veya universe değişince cached params geçerliliğini yitirir;
    o zaman bu fonksiyonla yeniden arama yapılmalı.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Panel-aware date-based walk-forward CV (tezde adil karşılaştırma için)
    if train_dates is not None:
        print(f"  [CV] Panel-aware date-based walk-forward (5 fold)")
        cv_folds = list(_make_panel_walk_forward_folds(train_dates, n_splits=5))
    else:
        from sklearn.model_selection import TimeSeriesSplit
        print(f"  [CV] ⚠️ train_dates verilmedi, TimeSeriesSplit kullanılıyor. "
              f"Panel data ile optimistik bias riski vardır.")
        tscv = TimeSeriesSplit(n_splits=5)
        cv_folds = list(tscv.split(X_train))

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
            'subsample': trial.suggest_float('subsample', 0.4, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, base_spw * 1.5),
        }

        cv_scores = []
        for train_idx, val_idx in cv_folds:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]


            model = xgb.XGBClassifier(
                **params,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1,
                tree_method='hist',
                device = 'cuda'
            )
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False,
            )

            probs = model.predict_proba(X_fold_val)[:, 1]
            try:
                score = average_precision_score(y_fold_val, probs)
            except ValueError:
                score = 0.0
            cv_scores.append(score)

        return np.mean(cv_scores)

    print(f"  Optuna Bayesian Search başlıyor ({n_trials} trial)...")
    study = optuna.create_study(direction='maximize', study_name='xgb_prauc')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_trial.params
    print(f"\n  ✓ Optuna Tamamlandı! En iyi CV PR-AUC: {study.best_value:.4f}")
    print(f"  ✓ En iyi parametreler:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"\n  💡 Bu parametreleri xgboost_model.py'deki CACHED_BEST_PARAMS'a "
          f"kopyalayıp gelecek çalıştırmalarda tekrar aramayı önleyebilirsin.")

    return best_params, study.best_value


def train_xgboost(X_train, y_train, X_val, y_val, feature_names, train_dates=None,
                  use_cached_params: bool = True):
    """
    XGBoost baseline modeli — Optuna hyperparameter search + final model + SHAP.

    Args:
        X_train, y_train: eğitim seti
        X_val, y_val: validation seti (final eval için)
        feature_names: feature isimleri (SHAP ve importance için)
        train_dates: Optional. Train set'in tarih indeksi (panel-aware CV için).
                     Verilmezse standart TimeSeriesSplit kullanılır (panel data
                     için optimistik bias riski vardır — uyarı verilir).
        use_cached_params: True (default) ise CACHED_BEST_PARAMS kullanılır,
                           Optuna arama atlanır (~30 saniye vs ~30 dakika).
                           False ise 100-trial Optuna search çalıştırılır.
                           Feature set veya universe değişirse False yapılmalı.

    Returns:
        final_model, best_params, (val_roc_auc, val_pr_auc)
    """
    print("\n[PHASE 3] XGBoost Baseline...")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    base_spw = neg / max(pos, 1)
    print(f"  Class Ratio → Stable: {neg:,} | Risk: {pos:,} | scale_pos_weight base: {base_spw:.1f}")

    # ═══════════════════════════════════════════════════════════════
    # ADIM 1: Hyperparameter seçimi
    # Ya cached params kullan (hızlı) ya da Optuna ile ara (yavaş)
    # ═══════════════════════════════════════════════════════════════
    if use_cached_params:
        print(f"\n  [PHASE 3.1] ⚡ Cached best params kullanılıyor "
              f"(önceki Optuna arama sonucu, CV PR-AUC: {CACHED_BEST_CV_PRAUC:.4f})")
        print(f"  Optuna arama atlanıyor — feature set değişmediyse bu güvenli.")
        best_params = dict(CACHED_BEST_PARAMS)
        best_cv_prauc = CACHED_BEST_CV_PRAUC
        for k, v in best_params.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6g}")
            else:
                print(f"    {k}: {v}")
    else:
        print(f"\n  [PHASE 3.1] 🔍 Optuna Bayesian Search çalıştırılıyor "
              f"(feature set değişmiş olabilir)...")
        best_params, best_cv_prauc = _run_optuna_search(
            X_train, y_train, train_dates, base_spw, n_trials=100
        )

    # ═══════════════════════════════════════════════════════════════
    # ADIM 2: Final Model Eğitimi (Early Stopping ile)
    # ═══════════════════════════════════════════════════════════════
    print("\n  [PHASE 3.2] Final model eğitiliyor (early_stopping_rounds=50)...")

    final_model = xgb.XGBClassifier(
        **best_params,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        tree_method='hist',
        device='cuda'
    )

    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=10,
    )

    best_iteration = final_model.best_iteration
    print(f"  ✓ Early stopping: Model {best_iteration} ağaçta durdu (max {best_params['n_estimators']})")

    # --- Learning Curve ---
    results = final_model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    os.makedirs(os.path.join('visualization', 'learning_curves'), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, results['validation_0']['logloss'], label='Train Loss (LogLoss)')
    plt.plot(x_axis, results['validation_1']['logloss'], label='Validation Loss (LogLoss)')
    if best_iteration < epochs:
        plt.axvline(x=best_iteration, color='red', linestyle='--', alpha=0.7,
                    label=f'Early Stop @ {best_iteration}')
    plt.legend()
    plt.title('XGBoost LogLoss Learning Curve')
    plt.ylabel('Log Loss')
    plt.xlabel('Trees (n_estimators)')
    plt.tight_layout()
    plt.savefig(os.path.join('visualization', 'learning_curves', 'xgboost_learning_curve.png'))
    plt.close()

    # --- Validation Metrics ---
    probs = final_model.predict_proba(X_val)[:, 1]
    try:
        xgb_val_roc_auc = roc_auc_score(y_val, probs)
        xgb_val_pr_auc = average_precision_score(y_val, probs)
    except ValueError:
        xgb_val_roc_auc = 0.5
        xgb_val_pr_auc = 0.0

    print(f"  ✓ Validation ROC-AUC: {xgb_val_roc_auc:.4f} | PR-AUC: {xgb_val_pr_auc:.4f}")

    # --- Optimal Threshold (MCC Maximization) ---
    # ÖNEMLİ: Transformer pipeline'ları da MCC-max eşik kullanıyor
    # (bkz. pytorch_trainer.find_best_threshold_mcc). Baseline ile ana modeli
    # AYNI kriterle optimize etmek "fair comparison" iddiası için şart.
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    from models.pytorch_trainer import find_best_threshold_mcc

    optimal_threshold, val_mcc_at_thresh = find_best_threshold_mcc(y_val, probs)

    print(f"  ✓ Optimal Threshold (MCC max): {optimal_threshold:.4f} "
          f"(val MCC={val_mcc_at_thresh:.4f})")

    y_pred = (probs >= optimal_threshold).astype(int)

    print("\n=== XGBoost Classification Report ===")
    print(classification_report(y_val, y_pred, target_names=['Stable (0)', 'Risk (1)'], zero_division=0))

    # Confusion Matrix
    os.makedirs(os.path.join('visualization', 'metrics'), exist_ok=True)
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stable', 'Risk'])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
    plt.title('XGBoost Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join('visualization', 'metrics', 'xgboost_confusion_matrix.png'))
    plt.close(fig)

    # --- Feature Importance (Gain) ---
    importances = final_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    os.makedirs('checkpoints', exist_ok=True)
    feat_imp_df.to_csv(os.path.join('checkpoints', 'feature_importances.csv'), index=False)

    print("\n=== Top 10 Feature Importances (Gain) ===")
    for _, row in feat_imp_df.head(10).iterrows():
        print(f"  {row['Feature']:<25}: {row['Importance']:.4f}")

    plt.figure(figsize=(10, 8))
    top_n = min(20, len(feature_names))
    plt.barh(feat_imp_df['Feature'].head(top_n)[::-1], feat_imp_df['Importance'].head(top_n)[::-1],
             color='skyblue')
    plt.title(f'XGBoost Top {top_n} Feature Importances (Gain)')
    plt.xlabel('Importance (Gain)')
    plt.tight_layout()
    plt.savefig(os.path.join('visualization', 'metrics', 'xgboost_feature_importances.png'))
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # ADIM 3: SHAP Analizi
    # ═══════════════════════════════════════════════════════════════
    print("\n  [PHASE 3.3] SHAP Analizi...")
    try:
        import shap

        explainer = shap.TreeExplainer(final_model)
        sample_size = min(2000, len(X_val))
        X_sample = X_val[:sample_size]
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            show=False,
            max_display=20
        )
        plt.tight_layout()
        plt.savefig(os.path.join('visualization', 'metrics', 'shap_summary.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ SHAP Summary Plot kaydedildi: visualization/metrics/shap_summary.png")

    except ImportError:
        print("  [WARNING] shap kütüphanesi bulunamadı, SHAP analizi atlanıyor.")
    except Exception as e:
        print(f"  [WARNING] SHAP analizi başarısız: {e}")

    # --- Save Model & Params ---
    final_model.save_model(os.path.join('checkpoints', 'xgboost_model.json'))

    with open(os.path.join('checkpoints', 'xgboost_eval.txt'), 'w') as f:
        f.write(f"Validation ROC-AUC: {xgb_val_roc_auc:.4f}\n")
        f.write(f"Validation PR-AUC: {xgb_val_pr_auc:.4f}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n")
        f.write(f"Optuna Best CV PR-AUC: {best_cv_prauc:.4f}\n\n")
        f.write("Best Params:\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")

    return final_model, best_params, (xgb_val_roc_auc, xgb_val_pr_auc)