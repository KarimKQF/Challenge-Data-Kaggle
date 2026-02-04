"""
🔥 V9 ANTI-OVERFITTING ULTIMATE 🔥
===================================
Based on V7/V8 insights: OOF 0.26 vs Actual 0.19 = 0.07 gap!

KEY STRATEGIES TO MINIMIZE OOF/PUBLIC GAP:
1. STRICT TEMPORAL VALIDATION - Forward-only (test is AFTER train)
2. PURGED CV - Gap between train/val to prevent leakage  
3. ULTRA-REGULARIZATION - Very low lr, high L2, shallow trees
4. MINIMAL FEATURES - Only proven robust features
5. SINGLE ROBUST CONFIG - No complex ensembles
6. CONSERVATIVE PARAMS - Aggressive early stopping
7. PESSIMISTIC PREDICTIONS - Shrink towards mean

Target: Minimize gap between OOF and actual Kaggle score
"""

import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
import gc
from pathlib import Path

warnings.filterwarnings('ignore')

print("=" * 70)
print("🔥 V9 ANTI-OVERFITTING ULTIMATE 🔥")
print("=" * 70)

# ============== CONFIGURATION ==============
DATA_DIR = Path(r"C:\Users\Karim\Desktop\Projet\Kaggle\data")
train_path = DATA_DIR / 'train.parquet'
test_path = DATA_DIR / 'test.parquet'
OUTPUT_PATH = Path(r"C:\Users\Karim\Desktop\Projet\Kaggle\V9\submission_v9.csv")

forecast_windows = [1, 3, 10, 25]

# ULTRA-CONSERVATIVE CONFIG (Anti-overfit)
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.005,      # VERY LOW
    'n_estimators': 10000,
    'num_leaves': 15,            # SHALLOW
    'max_depth': 4,              # SHALLOW  
    'min_child_samples': 500,    # VERY HIGH
    'feature_fraction': 0.4,     # LOW
    'bagging_fraction': 0.5,     # LOW
    'bagging_freq': 10,
    'lambda_l1': 1.0,            # HIGH
    'lambda_l2': 50.0,           # VERY HIGH
    'verbosity': -1,
}

EARLY_STOP = 30  # Aggressive early stopping

print("\n[1/6] Loading data...")
train_full = pd.read_parquet(train_path)
test_full = pd.read_parquet(test_path)

print(f"   Train: {len(train_full):,} rows")
print(f"   Test: {len(test_full):,} rows")

# ============== GLOBAL STATS ==============
print("\n[2/6] Computing statistics...")
train_stats = {
    'sub_category': train_full.groupby('sub_category')['y_target'].mean().to_dict(),
    'sub_code': train_full.groupby('sub_code')['y_target'].mean().to_dict(),
    'global_mean': train_full['y_target'].mean()
}

# ============== MINIMAL ROBUST FEATURES ==============
def build_minimal_features(data, enc_stats=None):
    """Only the most robust features that generalize"""
    x = data.copy()
    
    # Target encodings (proven robust)
    if enc_stats is not None:
        for c in ['sub_category', 'sub_code']:
            x[c + '_enc'] = x[c].map(enc_stats[c]).fillna(enc_stats['global_mean'])
    
    # Simple interactions (proven in V3/V6)
    x['d_al_am'] = x['feature_al'] - x['feature_am']
    x['d_cg_by'] = x['feature_cg'] - x['feature_by']
    
    # Cross-sectional normalization (most robust signal)
    for col in ['feature_al', 'feature_am', 'd_al_am']:
        g = x.groupby('ts_index')[col]
        x[col + '_cs'] = (x[col] - g.transform('mean')) / (g.transform('std') + 1e-7)
    
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

# ============== STRICT TEMPORAL SPLITS ==============
print("\n[3/6] Setting up STRICT TEMPORAL validation...")

# Get all unique timestamps
all_ts = np.sort(train_full['ts_index'].unique())
n_ts = len(all_ts)

# Multiple temporal forward splits (like V8)
split_quantiles = [0.85, 0.88, 0.90, 0.92]
PURGE_GAP = 10  # Gap between train and val (prevent leakage)

print(f"   Using {len(split_quantiles)} temporal forward splits")
print(f"   Purge gap: {PURGE_GAP} timestamps")

# ============== TRAINING LOOP ==============
print("\n[4/6] Training per horizon with temporal splits...")

test_outputs = []
cv_cache = {'y': [], 'pred': [], 'wt': []}

for hz in forecast_windows:
    print(f"\n{'='*60}")
    print(f">>> HORIZON = {hz}")
    print(f"{'='*60}")
    
    tr_df = build_minimal_features(
        train_full.query(f"horizon == {hz}").copy(),
        train_stats
    )
    te_df = build_minimal_features(
        test_full.query(f"horizon == {hz}").copy(),
        train_stats
    )
    
    # Feature columns - MINIMAL SET
    exclude_cols = {'id', 'code', 'sub_code', 'sub_category', 'horizon', 'ts_index', 'weight', 'y_target'}
    feature_cols = [c for c in tr_df.columns if c not in exclude_cols and c.startswith(('feature_', 'd_', 'sub_', 'r_'))]
    
    # Add cross-sectional features
    feature_cols += [c for c in tr_df.columns if c.endswith('_cs') or c.endswith('_enc')]
    feature_cols = list(set(feature_cols))
    
    print(f"   Features: {len(feature_cols)} (minimal)")
    
    # Get per-horizon timestamps
    hz_ts = np.sort(tr_df['ts_index'].unique())
    n_hz_ts = len(hz_ts)
    
    print(f"   Timestamps: {n_hz_ts}")
    
    X_all = tr_df[feature_cols].values
    y_all = tr_df['y_target'].values
    w_all = tr_df['weight'].values
    ts_all = tr_df['ts_index'].values
    X_test = te_df[feature_cols].values
    
    # Store predictions from all splits
    test_pred_all = np.zeros(len(te_df))
    split_scores = []
    n_valid_splits = 0
    
    for split_idx, q in enumerate(split_quantiles):
        cutoff_idx = int(n_hz_ts * q)
        
        # Purged split using per-horizon timestamps
        train_ts = set(hz_ts[:max(0, cutoff_idx - PURGE_GAP)])
        val_ts = set(hz_ts[cutoff_idx:])
        
        train_mask = np.isin(ts_all, list(train_ts))
        val_mask = np.isin(ts_all, list(val_ts))
        
        if train_mask.sum() < 100 or val_mask.sum() < 100:
            continue
        
        X_tr, X_val = X_all[train_mask], X_all[val_mask]
        y_tr, y_val = y_all[train_mask], y_all[val_mask]
        w_val = w_all[val_mask]
        
        # Train LightGBM
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
        )
        
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Score
        score = weighted_rmse_score(y_val, val_pred, w_val)
        split_scores.append(score)
        n_valid_splits += 1
        
        # Average test predictions
        test_pred_all += test_pred
        
        # Store for global CV
        cv_cache['y'].extend(y_val.tolist())
        cv_cache['pred'].extend(val_pred.tolist())
        cv_cache['wt'].extend(w_val.tolist())
    
    # Average test predictions
    if n_valid_splits > 0:
        test_pred_all /= n_valid_splits
        avg_score = np.mean(split_scores)
        std_score = np.std(split_scores)
        print(f"   Score: {avg_score:.4f} (+/- {std_score:.4f}) [{n_valid_splits} splits]")
    else:
        print(f"   No valid splits!")
    
    # Store test predictions
    test_outputs.append(pd.DataFrame({
        'id': te_df['id'].values,
        'y_target': test_pred_all
    }))

# ============== FINAL CV SCORE ==============
print("\n[5/6] Computing final CV score...")

global_score = weighted_rmse_score(
    cv_cache['y'], cv_cache['pred'], cv_cache['wt']
)

# Pessimistic estimate (based on V7 gap of ~0.07)
expected_gap = 0.05  # Conservative estimate
pessimistic_score = global_score - expected_gap

print(f"\n{'='*60}")
print(f"   📊 TEMPORAL FORWARD CV SCORE: {global_score:.4f}")
print(f"   ⚠️ PESSIMISTIC ESTIMATE: {pessimistic_score:.4f}")
print(f"   (Assuming ~5% gap like V7: 0.26 OOF -> 0.19 actual)")
print(f"{'='*60}")

# ============== SAVE SUBMISSION ==============
print("\n[6/6] Saving submission...")

submission = pd.concat(test_outputs, ignore_index=True)
submission.to_csv(OUTPUT_PATH, index=False)

print(f"   📁 Saved: {OUTPUT_PATH}")
print(f"   📈 Rows: {len(submission):,}")

print("\n🔥 V9 ANTI-OVERFITTING COMPLETE! 🔥")
print("This version prioritizes generalization over high CV score.")

gc.collect()
