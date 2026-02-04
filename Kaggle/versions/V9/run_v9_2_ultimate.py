"""
🔥 V9.2 ULTIMATE - V7 PROVEN + ANTI-OVERFIT 🔥
==============================================
Exact V7 approach that achieved 0.19 on Kaggle but with:
1. More regularization to reduce OOF gap
2. Fewer configs to reduce overfit
3. GPU acceleration
"""

import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
import gc
from sklearn.model_selection import GroupKFold
from pathlib import Path

warnings.filterwarnings('ignore')

print("=" * 70)
print("🔥 V9.2 ULTIMATE - V7 PROVEN + ANTI-OVERFIT 🔥")
print("=" * 70)

# ============== CONFIGURATION ==============
DATA_DIR = Path(r"C:\Users\Karim\Desktop\Projet\Kaggle\data")
train_path = DATA_DIR / 'train.parquet'
test_path = DATA_DIR / 'test.parquet'
OUTPUT_PATH = Path(r"C:\Users\Karim\Desktop\Projet\Kaggle\V9\submission_v9_2.csv")

forecast_windows = [1, 3, 10, 25]
N_FOLDS = 5  # Same as V7

# V7-style configs but with MORE REGULARIZATION
LGB_CONFIGS = [
    {   # Config 1: Conservative (like V7 but stronger L2)
        'objective': 'regression',
        'metric': 'rmse',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'learning_rate': 0.01,
        'n_estimators': 5000,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 400,     # Higher than V7
        'feature_fraction': 0.5,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'lambda_l1': 0.3,             # Higher than V7
        'lambda_l2': 25.0,            # Higher than V7
        'verbosity': -1,
    },
    {   # Config 2: Moderate (like V7 but stronger regularization)
        'objective': 'regression',
        'metric': 'rmse',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'learning_rate': 0.015,
        'n_estimators': 4000,
        'num_leaves': 50,             # Lower than V7
        'max_depth': 7,               # Lower than V7
        'min_child_samples': 300,     # Higher than V7
        'feature_fraction': 0.5,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'lambda_l1': 0.2,
        'lambda_l2': 20.0,            # Higher than V7
        'verbosity': -1,
    },
]

print("\n[1/5] Loading data...")
train_full = pd.read_parquet(train_path)
test_full = pd.read_parquet(test_path)

print(f"   Train: {len(train_full):,} rows")
print(f"   Test: {len(test_full):,} rows")

# ============== GLOBAL STATS ==============
print("\n[2/5] Computing statistics...")
train_stats = {
    'sub_category': train_full.groupby('sub_category')['y_target'].mean().to_dict(),
    'sub_code': train_full.groupby('sub_code')['y_target'].mean().to_dict(),
    'global_mean': train_full['y_target'].mean()
}

# ============== V7 FEATURES ==============
def build_robust_features(data, enc_stats=None):
    """Build robust features (exact V7 approach)"""
    x = data.copy()
    
    # Target encodings (robust)
    if enc_stats is not None:
        for c in ['sub_category', 'sub_code']:
            x[c + '_enc'] = x[c].map(enc_stats[c]).fillna(enc_stats['global_mean'])
    
    # Interaction features (proven in V7)
    x['d_al_am'] = x['feature_al'] - x['feature_am']
    x['r_al_am'] = x['feature_al'] / (x['feature_am'].abs() + 1e-7)
    x['d_cg_by'] = x['feature_cg'] - x['feature_by']
    x['r_cg_by'] = x['feature_cg'] / (x['feature_by'].abs() + 1e-7)
    x['sum_al_am'] = x['feature_al'] + x['feature_am']
    x['sum_cg_by'] = x['feature_cg'] + x['feature_by']
    
    # Cross-sectional normalization
    norm_cols = ['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'd_al_am', 'd_cg_by']
    for col in norm_cols:
        if col in x.columns:
            g = x.groupby('ts_index')[col]
            x[col + '_cs'] = (x[col] - g.transform('mean')) / (g.transform('std') + 1e-7)
    
    # Temporal features
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
    return float(np.sqrt(max(0.0, 1.0 - np.clip(ratio, 0.0, 1.0))))

# ============== TRAINING LOOP (V7 style) ==============
print(f"\n[3/5] Training with {N_FOLDS}-Fold CV + {len(LGB_CONFIGS)} configs...")

test_outputs = []
cv_cache = {'y': [], 'pred': [], 'wt': []}

for hz in forecast_windows:
    print(f"\n{'='*60}")
    print(f">>> HORIZON = {hz}")
    print(f"{'='*60}")
    
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
    groups = tr_df['ts_index'].values
    X_test = te_df[feature_cols].values
    
    # K-Fold CV
    oof_pred = np.zeros(len(tr_df))
    test_pred = np.zeros(len(te_df))
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        fold_test_pred = np.zeros(len(te_df))
        
        for cfg_idx, cfg in enumerate(LGB_CONFIGS):
            try:
                model = lgb.LGBMRegressor(**cfg)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(100, verbose=False)]
                )
            except Exception as e:
                # Fallback to CPU
                cfg_cpu = cfg.copy()
                cfg_cpu['device'] = 'cpu'
                del cfg_cpu['gpu_platform_id'], cfg_cpu['gpu_device_id']
                model = lgb.LGBMRegressor(**cfg_cpu)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(100, verbose=False)]
                )
            
            # Ensemble: average across configs
            oof_pred[val_idx] += model.predict(X_val) / len(LGB_CONFIGS)
            fold_test_pred += model.predict(X_test) / len(LGB_CONFIGS)
        
        test_pred += fold_test_pred / N_FOLDS
        
        # Print fold progress
        print(f"   Fold {fold+1}/{N_FOLDS} ✓", end='')
    
    print()
    
    # Compute horizon score
    hz_score = weighted_rmse_score(y, oof_pred, w)
    print(f"   OOF Score: {hz_score:.4f}")
    
    # Store for global CV
    cv_cache['y'].extend(y.tolist())
    cv_cache['pred'].extend(oof_pred.tolist())
    cv_cache['wt'].extend(w.tolist())
    
    # Store test predictions
    test_outputs.append(pd.DataFrame({
        'id': te_df['id'].values,
        'y_target': test_pred
    }))

# ============== FINAL CV SCORE ==============
print("\n[4/5] Computing final CV score...")

global_score = weighted_rmse_score(
    cv_cache['y'], cv_cache['pred'], cv_cache['wt']
)

# Estimate based on V7 gap: 0.26 -> 0.19 = ~27% relative drop
# With stronger regularization, expect smaller gap (~20%)
relative_gap = 0.20
pessimistic_score = global_score * (1 - relative_gap)

print(f"\n{'='*60}")
print(f"   📊 OOF WEIGHTED RMSE SCORE: {global_score:.4f}")
print(f"   🎯 V7 REFERENCE: 0.26 OOF -> 0.19 Kaggle")
print(f"   ⚠️ PESSIMISTIC ESTIMATE: {pessimistic_score:.4f}")
print(f"   (Expecting ~20% gap with stronger regularization)")
print(f"{'='*60}")

# ============== SAVE SUBMISSION ==============
print("\n[5/5] Saving submission...")

submission = pd.concat(test_outputs, ignore_index=True)
submission.to_csv(OUTPUT_PATH, index=False)

print(f"   📁 Saved: {OUTPUT_PATH}")
print(f"   📈 Rows: {len(submission):,}")

# Show prediction stats
print(f"\n   Prediction stats:")
print(f"      Mean: {submission['y_target'].mean():.6f}")
print(f"      Std:  {submission['y_target'].std():.6f}")
print(f"      Min:  {submission['y_target'].min():.6f}")
print(f"      Max:  {submission['y_target'].max():.6f}")

print("\n🔥 V9.2 ULTIMATE COMPLETE! 🔥")
print("Using V7 proven approach with stronger regularization.")

gc.collect()
