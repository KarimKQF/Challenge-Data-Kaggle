"""
V6 BALANCED - Simple & Robust (Anti-Overfitting)
=================================================
Strategy: V3 got 0.24 on Kaggle, V5 got 0.1866
-> Simpler models generalize better!

This version:
1. Uses EXACT same features as original V3 notebook
2. Stronger regularization to prevent overfitting
3. 3 seeds only (like winning notebooks)
4. Conservative hyperparameters
"""

import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
import gc

warnings.filterwarnings('ignore')

# ============== CONFIGURATION ==============
train_path = 'c:/Users/Karim/Desktop/Kaggle/data/train.parquet'
test_path = 'c:/Users/Karim/Desktop/Kaggle/data/test.parquet'
val_threshold = 3500
forecast_windows = [1, 3, 10, 25]

# CONSERVATIVE LightGBM config - strong regularization to prevent overfitting
lgb_cfg = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.015,
    'n_estimators': 4000,
    'num_leaves': 80,           # Same as V3
    'min_child_samples': 200,   # Same as V3
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 10.0,          # Strong L2 regularization
    'verbosity': -1,
}

# Only 3 seeds like V3 (2 seeds) but slightly more robust
SEEDS = [42, 2024, 7]

print("="*60)
print("V6 BALANCED - Simple & Robust")
print("="*60)

# ============== COMPUTE STATS (EXACTLY LIKE V3) ==============
print("\n[1/4] Computing Stats...")
temp = pd.read_parquet(
    train_path,
    columns=['sub_category', 'sub_code', 'y_target', 'ts_index']
)

train_only = temp[temp.ts_index <= val_threshold]

train_stats = {
    'sub_category': train_only.groupby('sub_category')['y_target'].mean().to_dict(),
    'sub_code': train_only.groupby('sub_code')['y_target'].mean().to_dict(),
    'global_mean': train_only['y_target'].mean()
}

del temp, train_only
gc.collect()

# ============== FEATURE ENGINEERING (EXACT V3 FEATURES) ==============
def build_context_features(data, enc_stats=None):
    """EXACT same features as the original V3 notebook - proven to work!"""
    x = data.copy()
    
    # Encoded categorical signals
    if enc_stats is not None:
        for c in ['sub_category', 'sub_code']:
            x[c + '_enc'] = x[c].map(enc_stats[c]).fillna(enc_stats['global_mean'])
    
    # Interaction features (EXACT same as V3)
    x['d_al_am'] = x['feature_al'] - x['feature_am']
    x['r_al_am'] = x['feature_al'] / (x['feature_am'] + 1e-7)
    x['d_cg_by'] = x['feature_cg'] - x['feature_by']
    
    # Cross-sectional normalization (EXACT same as V3)
    norm_cols = ['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'd_al_am']
    
    for col in norm_cols:
        g = x.groupby('ts_index')[col]
        x[col + '_cs'] = (x[col] - g.transform('mean')) / (g.transform('std') + 1e-7)
    
    # Temporal signal (EXACT same as V3)
    x['t_cycle'] = np.sin(2 * np.pi * x['ts_index'] / 100)
    
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

print(f"\n[2/4] Training with {len(SEEDS)} seeds (simple & robust)...")

for hz in forecast_windows:
    print(f"\n>>> Training horizon = {hz}")
    
    tr_df = build_context_features(
        pd.read_parquet(train_path).query(f"horizon == {hz}"),
        train_stats
    )
    te_df = build_context_features(
        pd.read_parquet(test_path).query(f"horizon == {hz}"),
        train_stats
    )
    
    feature_cols = [
        c for c in tr_df.columns
        if c not in {
            'id', 'code', 'sub_code', 'sub_category',
            'horizon', 'ts_index', 'weight', 'y_target'
        }
    ]
    
    print(f"   Features: {len(feature_cols)}")
    
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
    
    for i, seed in enumerate(SEEDS):
        print(f"   Seed {i+1}/{len(SEEDS)} ({seed})...", end=" ", flush=True)
        
        mdl = lgb.LGBMRegressor(**lgb_cfg, random_state=seed)
        
        mdl.fit(
            X_fit,
            y_fit,
            sample_weight=w_fit,
            eval_set=[(X_hold, y_hold)],
            eval_sample_weight=[w_hold],
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )
        
        val_pred += mdl.predict(X_hold) / len(SEEDS)
        tst_pred += mdl.predict(te_df[feature_cols]) / len(SEEDS)
        print(f"best_iter={mdl.best_iteration_}")
    
    cv_cache['y'].extend(y_hold.tolist())
    cv_cache['pred'].extend(val_pred.tolist())
    cv_cache['wt'].extend(w_hold.tolist())
    
    hz_score = weighted_rmse_score(y_hold, val_pred, w_hold)
    print(f"   >>> Horizon {hz} Score: {hz_score:.5f}")
    
    test_outputs.append(
        pd.DataFrame({'id': te_df['id'], 'prediction': tst_pred})
    )
    
    del tr_df, te_df
    gc.collect()

# ============== FINAL RESULTS ==============
print("\n[3/4] Computing final aggregate score...")
final_metric = weighted_rmse_score(cv_cache['y'], cv_cache['pred'], cv_cache['wt'])

print(f"\n{'='*60}")
print(f"   FINAL AGGREGATE SCORE: {final_metric:.6f}")
print(f"{'='*60}")

# ============== SAVE SUBMISSION ==============
print("\n[4/4] Saving submission...")
pd.concat(test_outputs).to_csv('submission.csv', index=False)
print("Submission saved: submission.csv")
print("\nV6 BALANCED COMPLETE! 🎯")
