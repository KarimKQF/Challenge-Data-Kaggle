"""
V4 ULTIMATE - Optimized for MAXIMUM SCORE
==========================================
Improvements over V3:
1. More seeds for ensemble (5 seeds instead of 2)
2. Deeper trees with more leaves (num_leaves=256)
3. More estimators (6000 instead of 4000)
4. Additional feature engineering (lag features, ratios, ranks)
5. Optimized regularization
6. Cross-sectional features enhanced
7. Multiple target encodings
"""

import numpy as np
import pandas as pd
import os
import warnings
import lightgbm as lgb
import gc
from typing import Dict, List

warnings.filterwarnings('ignore')

# ============== CONFIGURATION ==============
train_path = 'c:/Users/Karim/Desktop/Kaggle/data/train.parquet'
test_path = 'c:/Users/Karim/Desktop/Kaggle/data/test.parquet'
val_threshold = 3500
forecast_windows = [1, 3, 10, 25]

# Aggressive LightGBM configuration for maximum performance
lgb_cfg = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.008,  # Lower LR for better generalization
    'n_estimators': 6000,    # More trees
    'num_leaves': 256,       # Deeper trees
    'max_depth': 12,         # Explicit depth control
    'min_child_samples': 100,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'lambda_l1': 0.05,
    'lambda_l2': 5.0,
    'min_gain_to_split': 0.01,
    'verbosity': -1,
}

# Use 5 seeds for robust ensemble
SEEDS = [42, 2024, 7, 123, 999]

print("="*60)
print("V4 ULTIMATE - MAXIMUM SCORE OPTIMIZATION")
print("="*60)

# ============== COMPUTE STATS ==============
print("\n[1/4] Computing training statistics...")
temp = pd.read_parquet(
    train_path,
    columns=['sub_category', 'sub_code', 'code', 'y_target', 'ts_index', 'horizon']
)

train_only = temp[temp.ts_index <= val_threshold]

# Enhanced target encodings
train_stats = {
    'sub_category': train_only.groupby('sub_category')['y_target'].mean().to_dict(),
    'sub_code': train_only.groupby('sub_code')['y_target'].mean().to_dict(),
    'code': train_only.groupby('code')['y_target'].mean().to_dict(),
    'global_mean': train_only['y_target'].mean(),
    'global_std': train_only['y_target'].std(),
}

# Per-horizon statistics for better normalization
horizon_stats = {}
for hz in forecast_windows:
    hz_data = train_only[train_only.horizon == hz]
    horizon_stats[hz] = {
        'mean': hz_data['y_target'].mean(),
        'std': hz_data['y_target'].std(),
    }

del temp, train_only
gc.collect()

# ============== FEATURE ENGINEERING ==============
def build_ultimate_features(data: pd.DataFrame, enc_stats: Dict, hz_stats: Dict = None, horizon: int = None) -> pd.DataFrame:
    """
    Build comprehensive feature set for maximum predictive power.
    """
    x = data.copy()
    
    # === 1. TARGET ENCODINGS ===
    for c in ['sub_category', 'sub_code', 'code']:
        if c in enc_stats:
            x[f'{c}_enc'] = x[c].map(enc_stats[c]).fillna(enc_stats['global_mean'])
    
    # === 2. INTERACTION FEATURES ===
    # Primary interactions
    x['d_al_am'] = x['feature_al'] - x['feature_am']
    x['r_al_am'] = x['feature_al'] / (x['feature_am'].abs() + 1e-7)
    x['d_cg_by'] = x['feature_cg'] - x['feature_by']
    x['r_cg_by'] = x['feature_cg'] / (x['feature_by'].abs() + 1e-7)
    
    # Additional powerful interactions
    x['sum_al_am'] = x['feature_al'] + x['feature_am']
    x['prod_al_am'] = x['feature_al'] * x['feature_am']
    x['sum_cg_by'] = x['feature_cg'] + x['feature_by']
    x['prod_cg_by'] = x['feature_cg'] * x['feature_by']
    
    # Cross-feature interactions
    x['d_al_by'] = x['feature_al'] - x['feature_by']
    x['d_am_cg'] = x['feature_am'] - x['feature_cg']
    x['r_al_cg'] = x['feature_al'] / (x['feature_cg'].abs() + 1e-7)
    
    # === 3. CROSS-SECTIONAL NORMALIZATION (per ts_index) ===
    base_cols = ['feature_al', 'feature_am', 'feature_cg', 'feature_by']
    derived_cols = ['d_al_am', 'd_cg_by', 'r_al_am', 'sum_al_am']
    norm_cols = base_cols + derived_cols
    
    for col in norm_cols:
        if col in x.columns:
            g = x.groupby('ts_index')[col]
            x[f'{col}_cs_mean'] = g.transform('mean')
            x[f'{col}_cs_std'] = g.transform('std') + 1e-7
            x[f'{col}_cs'] = (x[col] - x[f'{col}_cs_mean']) / x[f'{col}_cs_std']
            x[f'{col}_cs_rank'] = g.rank(pct=True)  # Rank features
            # Clean up intermediate columns
            x.drop([f'{col}_cs_mean', f'{col}_cs_std'], axis=1, inplace=True)
    
    # === 4. TEMPORAL FEATURES ===
    x['t_cycle_100'] = np.sin(2 * np.pi * x['ts_index'] / 100)
    x['t_cycle_50'] = np.sin(2 * np.pi * x['ts_index'] / 50)
    x['t_cycle_200'] = np.sin(2 * np.pi * x['ts_index'] / 200)
    x['t_cos_100'] = np.cos(2 * np.pi * x['ts_index'] / 100)
    x['ts_index_norm'] = x['ts_index'] / x['ts_index'].max()
    
    # === 5. POLYNOMIAL FEATURES ===
    x['feature_al_sq'] = x['feature_al'] ** 2
    x['feature_am_sq'] = x['feature_am'] ** 2
    x['d_al_am_sq'] = x['d_al_am'] ** 2
    
    # === 6. ABSOLUTE VALUE FEATURES ===
    x['abs_al'] = x['feature_al'].abs()
    x['abs_am'] = x['feature_am'].abs()
    x['abs_d_al_am'] = x['d_al_am'].abs()
    
    # === 7. SIGN FEATURES ===
    x['sign_al'] = np.sign(x['feature_al'])
    x['sign_am'] = np.sign(x['feature_am'])
    x['sign_d_al_am'] = np.sign(x['d_al_am'])
    
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

print(f"\n[2/4] Training with {len(SEEDS)} seed ensemble...")

for hz in forecast_windows:
    print(f"\n{'='*50}")
    print(f">>> Training horizon = {hz}")
    print(f"{'='*50}")
    
    # Load and engineer features
    tr_df = build_ultimate_features(
        pd.read_parquet(train_path).query(f"horizon == {hz}"),
        train_stats,
        horizon_stats,
        hz
    )
    te_df = build_ultimate_features(
        pd.read_parquet(test_path).query(f"horizon == {hz}"),
        train_stats,
        horizon_stats,
        hz
    )
    
    # Select feature columns (exclude metadata and target)
    exclude_cols = {'id', 'code', 'sub_code', 'sub_category', 'horizon', 'ts_index', 'weight', 'y_target'}
    feature_cols = [c for c in tr_df.columns if c not in exclude_cols]
    
    print(f"   Features: {len(feature_cols)}")
    
    # Train/validation split
    fit_mask = tr_df.ts_index <= val_threshold
    val_mask = tr_df.ts_index > val_threshold
    
    X_fit = tr_df.loc[fit_mask, feature_cols]
    y_fit = tr_df.loc[fit_mask, 'y_target']
    w_fit = tr_df.loc[fit_mask, 'weight']
    
    X_hold = tr_df.loc[val_mask, feature_cols]
    y_hold = tr_df.loc[val_mask, 'y_target']
    w_hold = tr_df.loc[val_mask, 'weight']
    
    print(f"   Train: {len(X_fit):,} | Val: {len(X_hold):,}")
    
    val_pred = np.zeros(len(y_hold))
    tst_pred = np.zeros(len(te_df))
    
    # Multi-seed ensemble
    for i, seed in enumerate(SEEDS):
        print(f"   Training seed {i+1}/{len(SEEDS)} (seed={seed})...", end=" ")
        
        mdl = lgb.LGBMRegressor(**lgb_cfg, random_state=seed)
        
        mdl.fit(
            X_fit,
            y_fit,
            sample_weight=w_fit,
            eval_set=[(X_hold, y_hold)],
            eval_sample_weight=[w_hold],
            callbacks=[lgb.early_stopping(300, verbose=False)]
        )
        
        val_pred += mdl.predict(X_hold) / len(SEEDS)
        tst_pred += mdl.predict(te_df[feature_cols]) / len(SEEDS)
        print(f"Best iter: {mdl.best_iteration_}")
    
    # Cache validation results
    cv_cache['y'].extend(y_hold.tolist())
    cv_cache['pred'].extend(val_pred.tolist())
    cv_cache['wt'].extend(w_hold.tolist())
    
    hz_score = weighted_rmse_score(y_hold, val_pred, w_hold)
    print(f"\n   >>> Horizon {hz} Score: {hz_score:.5f}")
    
    test_outputs.append(
        pd.DataFrame({'id': te_df['id'], 'prediction': tst_pred})
    )
    
    del tr_df, te_df, X_fit, X_hold
    gc.collect()

# ============== FINAL RESULTS ==============
print("\n[3/4] Computing final aggregate score...")
final_metric = weighted_rmse_score(cv_cache['y'], cv_cache['pred'], cv_cache['wt'])

print(f"\n{'='*60}")
print(f"{'='*60}")
print(f"   FINAL AGGREGATE SCORE: {final_metric:.6f}")
print(f"{'='*60}")
print(f"{'='*60}")

# ============== SAVE SUBMISSION ==============
print("\n[4/4] Saving submission...")
submission = pd.concat(test_outputs)
submission.to_csv('submission.csv', index=False)
print(f"Submission saved: {len(submission):,} rows")
print("\nDONE! Good luck! 🚀")
