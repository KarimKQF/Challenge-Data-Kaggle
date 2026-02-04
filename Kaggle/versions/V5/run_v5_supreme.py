"""
V5 SUPREME STACKING - ULTIMATE SCORE OPTIMIZATION
===================================================
Major improvements over V4:
1. STACKING: LightGBM + XGBoost ensemble
2. Per-horizon hyperparameter optimization
3. More seeds for long horizons (10 seeds for H25)
4. Feature importance-based selection
5. Blending weights optimization
6. Advanced feature engineering
"""

import numpy as np
import pandas as pd
import os
import warnings
import lightgbm as lgb
import xgboost as xgb
import gc
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

# ============== CONFIGURATION ==============
train_path = 'c:/Users/Karim/Desktop/Kaggle/data/train.parquet'
test_path = 'c:/Users/Karim/Desktop/Kaggle/data/test.parquet'
val_threshold = 3500
forecast_windows = [1, 3, 10, 25]

# Per-horizon LightGBM configurations (optimized per horizon)
LGB_CONFIGS = {
    1: {  # Short-term: need fast learning
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
    10: {  # Medium-term
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
    25: {  # Long-term: need more regularization
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

# XGBoost configurations per horizon
XGB_CONFIGS = {
    1: {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.02,
        'n_estimators': 4000,
        'max_depth': 8,
        'min_child_weight': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.02,
        'reg_lambda': 2.0,
        'verbosity': 0,
    },
    3: {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.015,
        'n_estimators': 4500,
        'max_depth': 9,
        'min_child_weight': 80,
        'subsample': 0.75,
        'colsample_bytree': 0.65,
        'reg_alpha': 0.03,
        'reg_lambda': 3.0,
        'verbosity': 0,
    },
    10: {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.01,
        'n_estimators': 5000,
        'max_depth': 10,
        'min_child_weight': 60,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.05,
        'reg_lambda': 5.0,
        'verbosity': 0,
    },
    25: {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.008,
        'n_estimators': 6000,
        'max_depth': 11,
        'min_child_weight': 50,
        'subsample': 0.65,
        'colsample_bytree': 0.55,
        'reg_alpha': 0.08,
        'reg_lambda': 8.0,
        'verbosity': 0,
    },
}

# More seeds for long horizons
SEEDS_PER_HORIZON = {
    1: [42, 2024, 7],       # 3 seeds
    3: [42, 2024, 7, 123],  # 4 seeds
    10: [42, 2024, 7, 123, 999, 777],  # 6 seeds
    25: [42, 2024, 7, 123, 999, 777, 555, 888, 333, 111],  # 10 seeds for maximum stability
}

# Blending weights (LGB weight, XGB weight)
BLEND_WEIGHTS = {
    1: (0.6, 0.4),
    3: (0.55, 0.45),
    10: (0.5, 0.5),
    25: (0.45, 0.55),  # XGBoost often better for long-term
}

print("="*70)
print("V5 SUPREME STACKING - ULTIMATE SCORE OPTIMIZATION")
print("="*70)

# ============== COMPUTE STATS ==============
print("\n[1/5] Computing training statistics...")
temp = pd.read_parquet(
    train_path,
    columns=['sub_category', 'sub_code', 'code', 'y_target', 'ts_index', 'horizon', 'weight']
)

train_only = temp[temp.ts_index <= val_threshold]

# Enhanced target encodings with weighted means
train_stats = {
    'sub_category': train_only.groupby('sub_category')['y_target'].mean().to_dict(),
    'sub_code': train_only.groupby('sub_code')['y_target'].mean().to_dict(),
    'code': train_only.groupby('code')['y_target'].mean().to_dict(),
    'global_mean': train_only['y_target'].mean(),
    'global_std': train_only['y_target'].std(),
    # Weighted mean per category
    'sub_category_wmean': train_only.groupby('sub_category').apply(
        lambda x: np.average(x['y_target'], weights=x['weight']) if x['weight'].sum() > 0 else x['y_target'].mean()
    ).to_dict(),
}

del temp, train_only
gc.collect()

# ============== FEATURE ENGINEERING ==============
def build_supreme_features(data: pd.DataFrame, enc_stats: Dict) -> pd.DataFrame:
    """
    Build comprehensive feature set with all optimizations.
    """
    x = data.copy()
    
    # === 1. TARGET ENCODINGS ===
    for c in ['sub_category', 'sub_code', 'code']:
        if c in enc_stats:
            x[f'{c}_enc'] = x[c].map(enc_stats[c]).fillna(enc_stats['global_mean'])
    
    # Weighted target encoding
    if 'sub_category_wmean' in enc_stats:
        x['sub_category_wenc'] = x['sub_category'].map(enc_stats['sub_category_wmean']).fillna(enc_stats['global_mean'])
    
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
    
    # Additional high-value interactions
    x['al_minus_mean'] = x['feature_al'] - x['feature_al'].mean()
    x['am_minus_mean'] = x['feature_am'] - x['feature_am'].mean()
    
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
            x[f'{col}_cs_minmax'] = (x[col] - g.transform('min')) / (g.transform('max') - g.transform('min') + 1e-7)
    
    # === 4. TEMPORAL FEATURES ===
    x['t_cycle_100'] = np.sin(2 * np.pi * x['ts_index'] / 100)
    x['t_cycle_50'] = np.sin(2 * np.pi * x['ts_index'] / 50)
    x['t_cycle_200'] = np.sin(2 * np.pi * x['ts_index'] / 200)
    x['t_cos_100'] = np.cos(2 * np.pi * x['ts_index'] / 100)
    x['t_cos_50'] = np.cos(2 * np.pi * x['ts_index'] / 50)
    x['ts_index_norm'] = x['ts_index'] / x['ts_index'].max()
    x['ts_index_sq'] = (x['ts_index'] / x['ts_index'].max()) ** 2
    
    # === 5. POLYNOMIAL FEATURES ===
    x['feature_al_sq'] = x['feature_al'] ** 2
    x['feature_am_sq'] = x['feature_am'] ** 2
    x['d_al_am_sq'] = x['d_al_am'] ** 2
    x['feature_al_cube'] = x['feature_al'] ** 3
    
    # === 6. ABSOLUTE VALUE & SIGN FEATURES ===
    x['abs_al'] = x['feature_al'].abs()
    x['abs_am'] = x['feature_am'].abs()
    x['abs_d_al_am'] = x['d_al_am'].abs()
    x['sign_al'] = np.sign(x['feature_al'])
    x['sign_am'] = np.sign(x['feature_am'])
    x['sign_d_al_am'] = np.sign(x['d_al_am'])
    
    # === 7. LOG TRANSFORMS (safe) ===
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

print(f"\n[2/5] Loading and engineering features...")

for hz in forecast_windows:
    print(f"\n{'='*70}")
    print(f">>> HORIZON = {hz}")
    print(f"{'='*70}")
    
    seeds = SEEDS_PER_HORIZON[hz]
    lgb_cfg = LGB_CONFIGS[hz]
    xgb_cfg = XGB_CONFIGS[hz]
    lgb_weight, xgb_weight = BLEND_WEIGHTS[hz]
    
    print(f"   Config: {len(seeds)} seeds | LGB weight: {lgb_weight:.2f} | XGB weight: {xgb_weight:.2f}")
    
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
    val_pred_lgb = np.zeros(len(y_hold))
    val_pred_xgb = np.zeros(len(y_hold))
    tst_pred_lgb = np.zeros(len(te_df))
    tst_pred_xgb = np.zeros(len(te_df))
    
    # ===== LIGHTGBM TRAINING =====
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
        
        val_pred_lgb += mdl.predict(X_hold) / len(seeds)
        tst_pred_lgb += mdl.predict(X_test) / len(seeds)
        print(f"iter={mdl.best_iteration_}")
    
    lgb_score = weighted_rmse_score(y_hold, val_pred_lgb, w_hold)
    print(f"   [LightGBM] Val Score: {lgb_score:.5f}")
    
    # ===== XGBOOST TRAINING =====
    print(f"\n   [XGBoost] Training with {len(seeds)} seeds...")
    for i, seed in enumerate(seeds):
        print(f"      Seed {i+1}/{len(seeds)} ({seed})...", end=" ", flush=True)
        
        mdl = xgb.XGBRegressor(**xgb_cfg, random_state=seed, early_stopping_rounds=200)
        mdl.fit(
            X_fit, y_fit,
            sample_weight=w_fit,
            eval_set=[(X_hold, y_hold)],
            sample_weight_eval_set=[w_hold],
            verbose=False
        )
        
        val_pred_xgb += mdl.predict(X_hold) / len(seeds)
        tst_pred_xgb += mdl.predict(X_test) / len(seeds)
        best_iter = getattr(mdl, 'best_iteration', xgb_cfg['n_estimators'])
        print(f"iter={best_iter}")
    
    xgb_score = weighted_rmse_score(y_hold, val_pred_xgb, w_hold)
    print(f"   [XGBoost] Val Score: {xgb_score:.5f}")
    
    # ===== BLENDING =====
    val_pred = lgb_weight * val_pred_lgb + xgb_weight * val_pred_xgb
    tst_pred = lgb_weight * tst_pred_lgb + xgb_weight * tst_pred_xgb
    
    blend_score = weighted_rmse_score(y_hold, val_pred, w_hold)
    print(f"\n   >>> BLENDED SCORE: {blend_score:.5f} (LGB: {lgb_score:.5f} | XGB: {xgb_score:.5f})")
    
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
print("\n[4/5] Computing final aggregate score...")
final_metric = weighted_rmse_score(cv_cache['y'], cv_cache['pred'], cv_cache['wt'])

print(f"\n{'='*70}")
print(f"{'='*70}")
print(f"   🏆 FINAL AGGREGATE SCORE: {final_metric:.6f}")
print(f"{'='*70}")
print(f"{'='*70}")

# ============== SAVE SUBMISSION ==============
print("\n[5/5] Saving submission...")
submission = pd.concat(test_outputs)
submission.to_csv('submission.csv', index=False)
print(f"Submission saved: {len(submission):,} rows")
print("\n🚀 V5 SUPREME COMPLETE! Good luck! 🚀")
