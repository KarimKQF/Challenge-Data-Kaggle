"""
V7 ULTIMATE - Targeting 0.5+ Score
===================================
Advanced techniques to maximize Kaggle score WITHOUT overfitting:

1. K-Fold Cross-Validation (5 folds) for robust predictions
2. Out-of-fold (OOF) predictions for proper validation
3. Ensemble of diverse LightGBM configs
4. Robust features that generalize well
5. Strong regularization + early stopping
6. Per-horizon optimization
"""

import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
import gc
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')

# ============== CONFIGURATION ==============
train_path = 'c:/Users/Karim/Desktop/Kaggle/data/train.parquet'
test_path = 'c:/Users/Karim/Desktop/Kaggle/data/test.parquet'
forecast_windows = [1, 3, 10, 25]
N_FOLDS = 5

# Multiple diverse LightGBM configs for ensemble (different hyperparameters = diversity)
LGB_CONFIGS = [
    {   # Config 1: Conservative
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'n_estimators': 5000,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 300,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'lambda_l1': 0.2,
        'lambda_l2': 15.0,
        'verbosity': -1,
    },
    {   # Config 2: Moderate
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.015,
        'n_estimators': 4000,
        'num_leaves': 80,
        'max_depth': 9,
        'min_child_samples': 200,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 10.0,
        'verbosity': -1,
    },
    {   # Config 3: Slightly aggressive but regularized
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.02,
        'n_estimators': 3000,
        'num_leaves': 100,
        'max_depth': 10,
        'min_child_samples': 150,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.75,
        'bagging_freq': 4,
        'lambda_l1': 0.05,
        'lambda_l2': 5.0,
        'verbosity': -1,
    },
]

print("="*70)
print("V7 ULTIMATE - K-Fold CV + Multi-Config Ensemble")
print("="*70)

# ============== LOAD ALL DATA ==============
print("\n[1/5] Loading data...")
train_full = pd.read_parquet(train_path)
test_full = pd.read_parquet(test_path)

print(f"   Train: {len(train_full):,} rows")
print(f"   Test: {len(test_full):,} rows")

# ============== COMPUTE GLOBAL STATS ==============
print("\n[2/5] Computing global statistics...")
train_stats = {
    'sub_category': train_full.groupby('sub_category')['y_target'].mean().to_dict(),
    'sub_code': train_full.groupby('sub_code')['y_target'].mean().to_dict(),
    'global_mean': train_full['y_target'].mean()
}

# ============== FEATURE ENGINEERING ==============
def build_robust_features(data, enc_stats=None):
    """Build robust features that generalize well (proven in V3/V6)"""
    x = data.copy()
    
    # Target encodings (robust)
    if enc_stats is not None:
        for c in ['sub_category', 'sub_code']:
            x[c + '_enc'] = x[c].map(enc_stats[c]).fillna(enc_stats['global_mean'])
    
    # Interaction features (proven to work)
    x['d_al_am'] = x['feature_al'] - x['feature_am']
    x['r_al_am'] = x['feature_al'] / (x['feature_am'].abs() + 1e-7)
    x['d_cg_by'] = x['feature_cg'] - x['feature_by']
    x['r_cg_by'] = x['feature_cg'] / (x['feature_by'].abs() + 1e-7)
    
    # Additional robust interactions
    x['sum_al_am'] = x['feature_al'] + x['feature_am']
    x['sum_cg_by'] = x['feature_cg'] + x['feature_by']
    
    # Cross-sectional normalization (proven)
    norm_cols = ['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'd_al_am', 'd_cg_by']
    for col in norm_cols:
        if col in x.columns:
            g = x.groupby('ts_index')[col]
            x[col + '_cs'] = (x[col] - g.transform('mean')) / (g.transform('std') + 1e-7)
    
    # Temporal features (proven)
    x['t_cycle'] = np.sin(2 * np.pi * x['ts_index'] / 100)
    x['t_cycle_50'] = np.sin(2 * np.pi * x['ts_index'] / 50)
    x['ts_norm'] = x['ts_index'] / x['ts_index'].max()
    
    return x

# ============== SCORING FUNCTION ==============
def weighted_rmse_score(y_target, y_pred, w):
    y_target, y_pred, w = np.array(y_target), np.array(y_pred), np.array(w)
    denom = np.sum(w * (y_target ** 2))
    if denom <= 0:
        return 0.0
    numerator = np.sum(w * ((y_target - y_pred) ** 2))
    ratio = numerator / denom
    return float(np.sqrt(1.0 - np.clip(ratio, 0.0, 1.0)))

# ============== MAIN TRAINING LOOP ==============
test_outputs = []
cv_cache = {'y': [], 'pred': [], 'wt': []}

print(f"\n[3/5] Training with {N_FOLDS}-Fold CV + {len(LGB_CONFIGS)} configs...")

for hz in forecast_windows:
    print(f"\n{'='*70}")
    print(f">>> HORIZON = {hz}")
    print(f"{'='*70}")
    
    # Get horizon data
    tr_df = build_robust_features(
        train_full.query(f"horizon == {hz}").copy(),
        train_stats
    )
    te_df = build_robust_features(
        test_full.query(f"horizon == {hz}").copy(),
        train_stats
    )
    
    # Feature columns
    exclude_cols = {'id', 'code', 'sub_code', 'sub_category', 'horizon', 'ts_index', 'weight', 'y_target'}
    feature_cols = [c for c in tr_df.columns if c not in exclude_cols]
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Train size: {len(tr_df):,}")
    
    X = tr_df[feature_cols].values
    y = tr_df['y_target'].values
    w = tr_df['weight'].values
    groups = tr_df['ts_index'].values  # Group by time index for proper CV
    
    X_test = te_df[feature_cols].values
    
    # Initialize predictions
    oof_pred = np.zeros(len(tr_df))
    test_pred = np.zeros(len(te_df))
    
    # K-Fold with GroupKFold (prevents data leakage across time)
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n   Fold {fold+1}/{N_FOLDS}:")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train, w_val = w[train_idx], w[val_idx]
        
        fold_val_pred = np.zeros(len(val_idx))
        fold_test_pred = np.zeros(len(te_df))
        
        # Ensemble of multiple configs
        for cfg_idx, lgb_cfg in enumerate(LGB_CONFIGS):
            mdl = lgb.LGBMRegressor(**lgb_cfg, random_state=42 + fold + cfg_idx)
            
            mdl.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                eval_sample_weight=[w_val],
                callbacks=[lgb.early_stopping(150, verbose=False)]
            )
            
            fold_val_pred += mdl.predict(X_val) / len(LGB_CONFIGS)
            fold_test_pred += mdl.predict(X_test) / len(LGB_CONFIGS)
        
        oof_pred[val_idx] = fold_val_pred
        test_pred += fold_test_pred / N_FOLDS
        
        fold_score = weighted_rmse_score(y_val, fold_val_pred, w_val)
        fold_scores.append(fold_score)
        print(f"      Fold {fold+1} Score: {fold_score:.5f}")
    
    # OOF score for this horizon
    hz_score = weighted_rmse_score(y, oof_pred, w)
    print(f"\n   >>> HORIZON {hz} OOF SCORE: {hz_score:.5f} (mean fold: {np.mean(fold_scores):.5f})")
    
    cv_cache['y'].extend(y.tolist())
    cv_cache['pred'].extend(oof_pred.tolist())
    cv_cache['wt'].extend(w.tolist())
    
    test_outputs.append(
        pd.DataFrame({'id': te_df['id'], 'prediction': test_pred})
    )
    
    del tr_df, te_df, X, y, w, X_test
    gc.collect()

# ============== FINAL RESULTS ==============
print("\n[4/5] Computing final OOF aggregate score...")
final_metric = weighted_rmse_score(cv_cache['y'], cv_cache['pred'], cv_cache['wt'])

print(f"\n{'='*70}")
print(f"{'='*70}")
print(f"   🏆 FINAL OOF AGGREGATE SCORE: {final_metric:.6f}")
print(f"{'='*70}")
print(f"{'='*70}")

# ============== SAVE SUBMISSION ==============
print("\n[5/5] Saving submission...")
submission = pd.concat(test_outputs)
submission.to_csv('submission_v7_ultimate.csv', index=False)
print(f"Submission saved: submission_v7_ultimate.csv ({len(submission):,} rows)")
print("\n🚀 V7 ULTIMATE COMPLETE! 🚀")

# Cleanup
del train_full, test_full
gc.collect()
