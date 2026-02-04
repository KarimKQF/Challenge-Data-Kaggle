"""
🚀 V9.1 BALANCED ANTI-OVERFIT 🚀
=================================
Based on:
- V7: K-Fold + ensemble approach (0.26 OOF -> 0.19 Kaggle)
- V8: Split analysis at quantiles 0.896-0.898 (optimal temporal split point)

BALANCED STRATEGIES:
1. Use V8's optimal split quantile (~0.897)
2. Moderate regularization (not too aggressive)
3. Single temporal forward split (realistic test scenario)
4. Proven V7 features
5. GPU acceleration for speed
"""

import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
import gc
from pathlib import Path

warnings.filterwarnings('ignore')

print("=" * 70)
print("🚀 V9.1 BALANCED ANTI-OVERFIT 🚀")
print("=" * 70)

# ============== CONFIGURATION ==============
DATA_DIR = Path(r"C:\Users\Karim\Desktop\Projet\Kaggle\data")
train_path = DATA_DIR / 'train.parquet'
test_path = DATA_DIR / 'test.parquet'
OUTPUT_PATH = Path(r"C:\Users\Karim\Desktop\Projet\Kaggle\V9\submission_v9_1.csv")

forecast_windows = [1, 3, 10, 25]

# BALANCED CONFIG (based on V7 but more conservative)
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'learning_rate': 0.015,     # Moderate (V7 used 0.01-0.02)
    'n_estimators': 5000,
    'num_leaves': 31,           # Moderate
    'max_depth': 6,             # Moderate
    'min_child_samples': 200,   # Higher than V7 for anti-overfit
    'feature_fraction': 0.5,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
    'lambda_l1': 0.2,
    'lambda_l2': 15.0,          # High but not extreme
    'verbosity': -1,
}

EARLY_STOP = 100

# V8 optimal split point (from metrics_fine_split.txt analysis)
SPLIT_QUANTILE = 0.897  # Optimal point from V8 analysis

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

# ============== V7-STYLE ROBUST FEATURES ==============
def build_robust_features(data, enc_stats=None):
    """Build robust features (proven in V7)"""
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
    
    # Cross-sectional normalization (proven robust)
    norm_cols = ['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'd_al_am', 'd_cg_by']
    for col in norm_cols:
        if col in x.columns:
            g = x.groupby('ts_index')[col]
            x[col + '_cs'] = (x[col] - g.transform('mean')) / (g.transform('std') + 1e-7)
    
    # Temporal features (proven in V7)
    x['t_cycle'] = np.sin(2 * np.pi * x['ts_index'] / 100)
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

# ============== TRAINING LOOP ==============
print(f"\n[3/5] Training with V8 optimal split (q={SPLIT_QUANTILE})...")

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
    
    # Get per-horizon timestamps
    hz_ts = np.sort(tr_df['ts_index'].unique())
    n_hz_ts = len(hz_ts)
    
    # Use V8 optimal split
    cutoff_idx = int(n_hz_ts * SPLIT_QUANTILE)
    train_ts = set(hz_ts[:cutoff_idx])
    val_ts = set(hz_ts[cutoff_idx:])
    
    X_all = tr_df[feature_cols].values
    y_all = tr_df['y_target'].values
    w_all = tr_df['weight'].values
    ts_all = tr_df['ts_index'].values
    X_test = te_df[feature_cols].values
    
    train_mask = np.isin(ts_all, list(train_ts))
    val_mask = np.isin(ts_all, list(val_ts))
    
    X_tr, X_val = X_all[train_mask], X_all[val_mask]
    y_tr, y_val = y_all[train_mask], y_all[val_mask]
    w_val = w_all[val_mask]
    
    print(f"   Train: {train_mask.sum():,}, Val: {val_mask.sum():,}")
    
    # Train LightGBM
    try:
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
        )
    except Exception as e:
        # Fallback to CPU
        params_cpu = LGB_PARAMS.copy()
        params_cpu['device'] = 'cpu'
        del params_cpu['gpu_platform_id'], params_cpu['gpu_device_id']
        model = lgb.LGBMRegressor(**params_cpu)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
        )
    
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Score
    score = weighted_rmse_score(y_val, val_pred, w_val)
    print(f"   Val Score: {score:.4f}")
    
    # Store for global CV
    cv_cache['y'].extend(y_val.tolist())
    cv_cache['pred'].extend(val_pred.tolist())
    cv_cache['wt'].extend(w_val.tolist())
    
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

# Estimate based on V7 gap (26% -> 19% = ~27% relative drop)
relative_gap = 0.27
pessimistic_score = global_score * (1 - relative_gap)

print(f"\n{'='*60}")
print(f"   📊 TEMPORAL CV SCORE (q={SPLIT_QUANTILE}): {global_score:.4f}")
print(f"   ⚠️ PESSIMISTIC ESTIMATE: {pessimistic_score:.4f}")
print(f"   (Based on V7 relative gap of ~27%)")
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

print("\n🚀 V9.1 BALANCED COMPLETE! 🚀")

gc.collect()
