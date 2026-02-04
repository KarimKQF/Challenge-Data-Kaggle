"""
V5 FIXED - LightGBM SUPREME (No XGBoost - it wasn't working)
=============================================================
Based on V5 results, LightGBM was performing well:
- H1: 0.09409
- H10: 0.24543  
- H25: 0.28833

This version uses only LightGBM with:
1. Per-horizon optimized configs
2. More seeds (10 for long horizons)
3. 143 engineered features
4. Removed slow weighted groupby
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

# Per-horizon LightGBM configurations (optimized)
LGB_CONFIGS = {
    1: {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.02,
        'n_estimators': 5000,
        'num_leaves': 127,
        'max_depth': 10,
        'min_child_samples': 150,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 3,
        'lambda_l1': 0.02,
        'lambda_l2': 2.0,
        'verbosity': -1,
    },
    3: {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.015,
        'n_estimators': 5000,
        'num_leaves': 200,
        'max_depth': 11,
        'min_child_samples': 120,
        'feature_fraction': 0.65,
        'bagging_fraction': 0.75,
        'bagging_freq': 4,
        'lambda_l1': 0.03,
        'lambda_l2': 3.0,
        'verbosity': -1,
    },
    10: {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'n_estimators': 6000,
        'num_leaves': 256,
        'max_depth': 12,
        'min_child_samples': 100,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 0.05,
        'lambda_l2': 5.0,
        'verbosity': -1,
    },
    25: {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.008,
        'n_estimators': 7000,
        'num_leaves': 300,
        'max_depth': 14,
        'min_child_samples': 80,
        'feature_fraction': 0.55,
        'bagging_fraction': 0.65,
        'bagging_freq': 5,
        'lambda_l1': 0.08,
        'lambda_l2': 8.0,
        'verbosity': -1,
    },
}

# More seeds for longer horizons (for stability)
SEEDS_PER_HORIZON = {
    1: [42, 2024, 7, 123],           # 4 seeds
    3: [42, 2024, 7, 123, 999],      # 5 seeds
    10: [42, 2024, 7, 123, 999, 777, 555],  # 7 seeds
    25: [42, 2024, 7, 123, 999, 777, 555, 888, 333, 111],  # 10 seeds
}

print("="*70)
print("V5 FIXED - LightGBM SUPREME")
print("="*70)

# ============== COMPUTE STATS ==============
print("\n[1/4] Computing training statistics...")
temp = pd.read_parquet(
    train_path,
    columns=['sub_category', 'sub_code', 'code', 'y_target', 'ts_index']
)

train_only = temp[temp.ts_index <= val_threshold]

train_stats = {
    'sub_category': train_only.groupby('sub_category')['y_target'].mean().to_dict(),
    'sub_code': train_only.groupby('sub_code')['y_target'].mean().to_dict(),
    'code': train_only.groupby('code')['y_target'].mean().to_dict(),
    'global_mean': train_only['y_target'].mean(),
    'global_std': train_only['y_target'].std(),
}

del temp, train_only
gc.collect()

# ============== FEATURE ENGINEERING ==============
def build_supreme_features(data: pd.DataFrame, enc_stats: Dict) -> pd.DataFrame:
    x = data.copy()
    
    # === 1. TARGET ENCODINGS ===
    for c in ['sub_category', 'sub_code', 'code']:
        if c in enc_stats:
            x[f'{c}_enc'] = x[c].map(enc_stats[c]).fillna(enc_stats['global_mean'])
    
    # === 2. INTERACTION FEATURES ===
    x['d_al_am'] = x['feature_al'] - x['feature_am']
    x['r_al_am'] = x['feature_al'] / (x['feature_am'].abs() + 1e-7)
    x['d_cg_by'] = x['feature_cg'] - x['feature_by']
    x['r_cg_by'] = x['feature_cg'] / (x['feature_by'].abs() + 1e-7)
    x['sum_al_am'] = x['feature_al'] + x['feature_am']
    x['prod_al_am'] = x['feature_al'] * x['feature_am']
    x['sum_cg_by'] = x['feature_cg'] + x['feature_by']
    x['prod_cg_by'] = x['feature_cg'] * x['feature_by']
    x['d_al_by'] = x['feature_al'] - x['feature_by']
    x['d_am_cg'] = x['feature_am'] - x['feature_cg']
    x['r_al_cg'] = x['feature_al'] / (x['feature_cg'].abs() + 1e-7)
    
    # === 3. CROSS-SECTIONAL NORMALIZATION ===
    base_cols = ['feature_al', 'feature_am', 'feature_cg', 'feature_by']
    derived_cols = ['d_al_am', 'd_cg_by', 'sum_al_am']
    norm_cols = base_cols + derived_cols
    
    for col in norm_cols:
        if col in x.columns:
            g = x.groupby('ts_index')[col]
            cs_mean = g.transform('mean')
            cs_std = g.transform('std') + 1e-7
            x[f'{col}_cs'] = (x[col] - cs_mean) / cs_std
            x[f'{col}_cs_rank'] = g.rank(pct=True)
    
    # === 4. TEMPORAL FEATURES ===
    x['t_cycle_100'] = np.sin(2 * np.pi * x['ts_index'] / 100)
    x['t_cycle_50'] = np.sin(2 * np.pi * x['ts_index'] / 50)
    x['t_cycle_200'] = np.sin(2 * np.pi * x['ts_index'] / 200)
    x['t_cos_100'] = np.cos(2 * np.pi * x['ts_index'] / 100)
    x['t_cos_50'] = np.cos(2 * np.pi * x['ts_index'] / 50)
    x['ts_index_norm'] = x['ts_index'] / x['ts_index'].max()
    
    # === 5. POLYNOMIAL FEATURES ===
    x['feature_al_sq'] = x['feature_al'] ** 2
    x['feature_am_sq'] = x['feature_am'] ** 2
    x['d_al_am_sq'] = x['d_al_am'] ** 2
    
    # === 6. ABSOLUTE VALUE & SIGN FEATURES ===
    x['abs_al'] = x['feature_al'].abs()
    x['abs_am'] = x['feature_am'].abs()
    x['abs_d_al_am'] = x['d_al_am'].abs()
    x['sign_al'] = np.sign(x['feature_al'])
    x['sign_am'] = np.sign(x['feature_am'])
    x['sign_d_al_am'] = np.sign(x['d_al_am'])
    
    # === 7. LOG TRANSFORMS ===
    x['log_abs_al'] = np.log1p(x['feature_al'].abs())
    x['log_abs_am'] = np.log1p(x['feature_am'].abs())
    
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

print(f"\n[2/4] Training LightGBM models...")

for hz in forecast_windows:
    print(f"\n{'='*70}")
    print(f">>> HORIZON = {hz}")
    print(f"{'='*70}")
    
    seeds = SEEDS_PER_HORIZON[hz]
    lgb_cfg = LGB_CONFIGS[hz]
    
    print(f"   Config: {len(seeds)} seeds")
    
    # Load and engineer features
    print(f"   Loading data...")
    tr_df = build_supreme_features(
        pd.read_parquet(train_path).query(f"horizon == {hz}"),
        train_stats
    )
    te_df = build_supreme_features(
        pd.read_parquet(test_path).query(f"horizon == {hz}"),
        train_stats
    )
    
    # Select feature columns
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
    
    X_test = te_df[feature_cols]
    
    print(f"   Train: {len(X_fit):,} | Val: {len(X_hold):,} | Test: {len(X_test):,}")
    
    # Initialize predictions
    val_pred = np.zeros(len(y_hold))
    tst_pred = np.zeros(len(te_df))
    
    # Multi-seed LightGBM training
    print(f"\n   [LightGBM] Training with {len(seeds)} seeds...")
    for i, seed in enumerate(seeds):
        print(f"      Seed {i+1}/{len(seeds)} ({seed})...", end=" ", flush=True)
        
        mdl = lgb.LGBMRegressor(**lgb_cfg, random_state=seed)
        mdl.fit(
            X_fit, y_fit,
            sample_weight=w_fit,
            eval_set=[(X_hold, y_hold)],
            eval_sample_weight=[w_hold],
            callbacks=[lgb.early_stopping(250, verbose=False)]
        )
        
        val_pred += mdl.predict(X_hold) / len(seeds)
        tst_pred += mdl.predict(X_test) / len(seeds)
        print(f"iter={mdl.best_iteration_}")
    
    hz_score = weighted_rmse_score(y_hold, val_pred, w_hold)
    print(f"\n   >>> HORIZON {hz} SCORE: {hz_score:.5f}")
    
    # Cache results
    cv_cache['y'].extend(y_hold.tolist())
    cv_cache['pred'].extend(val_pred.tolist())
    cv_cache['wt'].extend(w_hold.tolist())
    
    test_outputs.append(
        pd.DataFrame({'id': te_df['id'], 'prediction': tst_pred})
    )
    
    del tr_df, te_df, X_fit, X_hold, X_test
    gc.collect()

# ============== FINAL RESULTS ==============
print("\n[3/4] Computing final aggregate score...")
final_metric = weighted_rmse_score(cv_cache['y'], cv_cache['pred'], cv_cache['wt'])

print(f"\n{'='*70}")
print(f"{'='*70}")
print(f"   🏆 FINAL AGGREGATE SCORE: {final_metric:.6f}")
print(f"{'='*70}")
print(f"{'='*70}")

# ============== SAVE SUBMISSION ==============
print("\n[4/4] Saving submission...")
submission = pd.concat(test_outputs)
submission.to_csv('submission_v5_fixed.csv', index=False)
print(f"Submission saved: submission_v5_fixed.csv ({len(submission):,} rows)")
print("\n🚀 V5 FIXED COMPLETE! Good luck! 🚀")
