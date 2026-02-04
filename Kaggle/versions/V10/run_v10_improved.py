"""
V10 IMPROVED - Based on V7 Success (0.1952)
============================================
Improvements over V7:
1. Seed Averaging (3 seeds) for more stable predictions
2. Slightly more regularization to reduce overfitting gap
3. Same proven feature engineering
4. GPU acceleration for speed
5. Correct output format (prediction column)

Target: Beat 0.1952
"""

import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
import gc
from sklearn.model_selection import GroupKFold
from pathlib import Path

warnings.filterwarnings('ignore')

# ============== CONFIGURATION ==============
DATA_DIR = Path(r"C:\Users\Karim\Desktop\PROJET LABS\Kaggle\data")
train_path = DATA_DIR / 'train.parquet'
test_path = DATA_DIR / 'test.parquet'
OUTPUT_PATH = Path(r"C:\Users\Karim\Desktop\PROJET LABS\Kaggle\V10\submission_v10.csv")

forecast_windows = [1, 3, 10, 25]
N_FOLDS = 5
N_SEEDS = 3  # Seed averaging for stability
SEEDS = [42, 2024, 1337]

# LightGBM configs - slightly more regularized than V7
LGB_CONFIGS = [
    {   # Config 1: Conservative (slightly more regularized)
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'n_estimators': 5000,
        'num_leaves': 55,       # Reduced from 63
        'max_depth': 7,         # Reduced from 8
        'min_child_samples': 350,   # Increased from 300
        'feature_fraction': 0.5,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'lambda_l1': 0.3,       # Increased from 0.2
        'lambda_l2': 20.0,      # Increased from 15
        'verbosity': -1,
    },
    {   # Config 2: Moderate
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.012,     # Slightly lower
        'n_estimators': 4500,
        'num_leaves': 70,           # Reduced from 80
        'max_depth': 8,             # Reduced from 9
        'min_child_samples': 250,   # Increased from 200
        'feature_fraction': 0.55,
        'bagging_fraction': 0.65,
        'bagging_freq': 5,
        'lambda_l1': 0.15,
        'lambda_l2': 12.0,
        'verbosity': -1,
    },
]

print("="*70)
print("🚀 V10 IMPROVED - Based on V7 Success 🚀")
print("="*70)
print(f"   Configs: {len(LGB_CONFIGS)}")
print(f"   Folds: {N_FOLDS}")
print(f"   Seeds: {N_SEEDS}")

# ============== LOAD DATA ==============
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

# ============== FEATURE ENGINEERING (same as V7) ==============
def build_robust_features(data, enc_stats=None):
    """Build robust features that generalize well (proven in V7)"""
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

print(f"\n[3/5] Training with {N_FOLDS}-Fold CV + {len(LGB_CONFIGS)} configs + {N_SEEDS} seeds...")

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
        
        n_models = len(LGB_CONFIGS) * N_SEEDS
        
        # Ensemble of multiple configs x seeds
        for cfg_idx, lgb_cfg in enumerate(LGB_CONFIGS):
            for seed_idx, seed in enumerate(SEEDS):
                mdl = lgb.LGBMRegressor(**lgb_cfg, random_state=seed + fold)
                
                mdl.fit(
                    X_train, y_train,
                    sample_weight=w_train,
                    eval_set=[(X_val, y_val)],
                    eval_sample_weight=[w_val],
                    callbacks=[lgb.early_stopping(150, verbose=False)]
                )
                
                fold_val_pred += mdl.predict(X_val) / n_models
                fold_test_pred += mdl.predict(X_test) / n_models
        
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
    
    # IMPORTANT: Use 'prediction' column for Kaggle (not y_target!)
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
print(f"   📊 V7 Reference: 0.1952")
print(f"   📉 Expected Kaggle (pessimistic): {final_metric * 0.73:.4f}")
print(f"{'='*70}")
print(f"{'='*70}")

# ============== SAVE SUBMISSION ==============
print("\n[5/5] Saving submission...")
submission = pd.concat(test_outputs, ignore_index=True)

# Validate submission
print(f"\n   Validation:")
print(f"      Shape: {submission.shape}")
print(f"      Columns: {submission.columns.tolist()}")
print(f"      Prediction mean: {submission['prediction'].mean():.6f}")
print(f"      Prediction std: {submission['prediction'].std():.6f}")
print(f"      NaN count: {submission['prediction'].isna().sum()}")

submission.to_csv(OUTPUT_PATH, index=False)
print(f"\n   📁 Saved: {OUTPUT_PATH}")
print(f"   📈 Rows: {len(submission):,}")

print("\n🚀 V10 IMPROVED COMPLETE! 🚀")

# Cleanup
del train_full, test_full
gc.collect()
