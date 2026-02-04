"""
🔥 QRT V5 ULTIMATE - NOUVELLE APPROCHE TOTALE 🔥
=================================================
Objectif: Battre 51.18% avec une approche radicalement différente

Stratégie V5:
1. XGBoost GPU + CatBoost (PAS LightGBM)
2. Features CROSS-SECTIONNELLES (ranking par timestamp)
3. Features de MOMENTUM AVANCÉ
4. STACKING des prédictions
5. Threshold OPTIMIZATION
6. Seed averaging massif
"""

import numpy as np
import pandas as pd
import warnings
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
import gc

warnings.filterwarnings('ignore')

# ============== CONFIGURATION ==============
DATA_DIR = 'c:/Users/Karim/Desktop/PROJET LABS/QRT/data/'
X_TRAIN_PATH = DATA_DIR + 'X_train_9xQjqvZ.csv'
Y_TRAIN_PATH = DATA_DIR + 'y_train_Ppwhaz8.csv'
X_TEST_PATH = DATA_DIR + 'X_test_1zTtEnD.csv'
SAMPLE_SUB_PATH = DATA_DIR + 'sample_submission_SpGVFuH.csv'
OUTPUT_PATH = 'c:/Users/Karim/Desktop/PROJET LABS/QRT/V5/submission.csv'

N_FOLDS = 5  # Moins de folds = plus de données train
N_SEEDS = 5  # Plus de seeds = plus stable
BASE_SEED = 42

print("=" * 70)
print("🔥 QRT V5 ULTIMATE - NOUVELLE APPROCHE TOTALE 🔥")
print("=" * 70)

# ============== LOAD DATA ==============
print("\n[1/6] Chargement des données...")
X_train = pd.read_csv(X_TRAIN_PATH, index_col='ROW_ID')
y_train = pd.read_csv(Y_TRAIN_PATH, index_col='ROW_ID')
X_test = pd.read_csv(X_TEST_PATH, index_col='ROW_ID')
sample_submission = pd.read_csv(SAMPLE_SUB_PATH, index_col='ROW_ID')

print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_test: {X_test.shape}")

# ============== NOUVELLE FEATURE ENGINEERING ==============
print("\n[2/6] Feature Engineering AVANCÉ (cross-sectionnel)...")

RET_features = [f'RET_{i}' for i in range(1, 21)]
VOL_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]

def advanced_features(df):
    """
    Features CROSS-SECTIONNELLES - Classement RELATIF par timestamp
    C'est ça qui fait la différence!
    """
    data = df.copy()
    
    # === ROLLING STATS (base) ===
    for h in [3, 5, 10, 20]:
        data[f'RET_MEAN_{h}'] = data[RET_features[:h]].mean(axis=1)
        data[f'RET_STD_{h}'] = data[RET_features[:h]].std(axis=1)
        data[f'RET_SUM_{h}'] = data[RET_features[:h]].sum(axis=1)
    
    # === MOMENTUM (prouvé efficace) ===
    data['MOM_SHORT'] = data[RET_features[:5]].sum(axis=1)
    data['MOM_LONG'] = data[RET_features[:20]].sum(axis=1)
    data['MOM_DIFF'] = data['MOM_SHORT'] - data['MOM_LONG'] / 4
    
    # === VOLUME ===
    data['VOL_SHORT'] = data[VOL_features[:5]].mean(axis=1)
    data['VOL_LONG'] = data[VOL_features[:20]].mean(axis=1)
    
    # === SHARPE-LIKE ===
    data['SHARPE_20'] = data['RET_MEAN_20'] / (data['RET_STD_20'] + 1e-8)
    
    # === CROSS-SECTIONAL (RANKING PAR TIMESTAMP) ===
    # C'est ÇA le secret - rank parmi tous les actifs du même timestamp
    for col in ['RET_MEAN_5', 'RET_MEAN_20', 'MOM_SHORT', 'MOM_DIFF', 'SHARPE_20', 'MEDIAN_DAILY_TURNOVER']:
        if col in data.columns:
            # Percentile rank (0-1)
            data[f'{col}_RANK'] = data.groupby('TS')[col].rank(pct=True)
            # Z-score normalisé par timestamp
            g = data.groupby('TS')[col]
            data[f'{col}_ZSCORE'] = (data[col] - g.transform('mean')) / (g.transform('std') + 1e-8)
    
    # === INTERACTION FEATURES ===
    data['MOM_VOL'] = data['MOM_SHORT'] * data['VOL_SHORT']
    data['MOM_RANK_VOL_RANK'] = data['MOM_SHORT_RANK'] * data['RET_MEAN_5_RANK']
    
    # === SIGN FEATURES ===
    data['POS_RATIO_5'] = (data[RET_features[:5]] > 0).sum(axis=1) / 5
    data['POS_RATIO_20'] = (data[RET_features[:20]] > 0).sum(axis=1) / 20
    
    # === LAST RETURN (très prédictif) ===
    data['RET_1_SIGN'] = (data['RET_1'] > 0).astype(int)
    data['RET_1_ABS'] = data['RET_1'].abs()
    
    return data

X_train = advanced_features(X_train)
X_test = advanced_features(X_test)

# ============== PREPARE FEATURES ==============
print("\n[3/6] Préparation des features...")

# Features sélectionnées
exclude_cols = {'TS', 'ALLOCATION', 'GROUP'}
feature_cols = [c for c in X_train.columns if c not in exclude_cols]
features = [c for c in feature_cols if X_train[c].dtype in ['float64', 'int64', 'float32', 'int32', 'int8']]

print(f"   Total features: {len(features)}")

X = X_train[features].fillna(0).values.astype(np.float32)
y = y_train['target'].values.astype(np.float32)
X_te = X_test[features].fillna(0).values.astype(np.float32)

train_dates = X_train['TS'].unique()

# ============== MODEL CONFIGS ==============
print("\n[4/6] Configuration modèles XGBoost + CatBoost...")

# XGBoost GPU config (différent de LightGBM!)
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'device': 'cuda',
    'learning_rate': 0.015,
    'max_depth': 5,
    'min_child_weight': 50,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.05,
    'reg_lambda': 2.0,
    'n_estimators': 800,
    'verbosity': 0,
}

# CatBoost config
cat_params = {
    'loss_function': 'RMSE',
    'task_type': 'GPU',
    'devices': '0',
    'learning_rate': 0.02,
    'depth': 6,
    'l2_leaf_reg': 5,
    'iterations': 800,
    'verbose': False,
    'early_stopping_rounds': 100,
}

# ============== TRAINING AVEC SEED AVERAGING ==============
print(f"\n[5/6] Training {N_FOLDS}-Fold CV x {N_SEEDS} Seeds...")

oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
test_xgb = np.zeros(len(X_te))
test_cat = np.zeros(len(X_te))

for seed_idx in range(N_SEEDS):
    seed = BASE_SEED + seed_idx * 100
    print(f"\n   === SEED {seed_idx + 1}/{N_SEEDS} (seed={seed}) ===")
    
    kf = KFold(n_splits=N_FOLDS, random_state=seed, shuffle=True)
    
    for fold, (train_date_idx, val_date_idx) in enumerate(kf.split(train_dates)):
        local_train_dates = train_dates[train_date_idx]
        local_val_dates = train_dates[val_date_idx]
        
        train_mask = X_train['TS'].isin(local_train_dates)
        val_mask = X_train['TS'].isin(local_val_dates)
        
        X_tr = X_train.loc[train_mask, features].fillna(0).values
        y_tr = y_train.loc[train_mask, 'target'].values
        X_val = X_train.loc[val_mask, features].fillna(0).values
        y_val = y_train.loc[val_mask, 'target'].values
        
        # === XGBoost ===
        try:
            xgb_cfg = xgb_params.copy()
            xgb_cfg['random_state'] = seed
            xgb_model = xgb.XGBRegressor(**xgb_cfg)
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            oof_xgb[val_mask.values] += xgb_model.predict(X_val) / (N_SEEDS)
            test_xgb += xgb_model.predict(X_te) / (N_SEEDS * N_FOLDS)
        except Exception as e:
            # CPU fallback
            xgb_cfg['tree_method'] = 'hist'
            xgb_cfg['device'] = 'cpu'
            xgb_model = xgb.XGBRegressor(**xgb_cfg)
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            oof_xgb[val_mask.values] += xgb_model.predict(X_val) / (N_SEEDS)
            test_xgb += xgb_model.predict(X_te) / (N_SEEDS * N_FOLDS)
        
        # === CatBoost ===
        try:
            cat_cfg = cat_params.copy()
            cat_cfg['random_seed'] = seed
            cat_model = CatBoostRegressor(**cat_cfg)
            cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            oof_cat[val_mask.values] += cat_model.predict(X_val) / (N_SEEDS)
            test_cat += cat_model.predict(X_te) / (N_SEEDS * N_FOLDS)
        except Exception as e:
            cat_cfg['task_type'] = 'CPU'
            del cat_cfg['devices']
            cat_model = CatBoostRegressor(**cat_cfg)
            cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            oof_cat[val_mask.values] += cat_model.predict(X_val) / (N_SEEDS)
            test_cat += cat_model.predict(X_te) / (N_SEEDS * N_FOLDS)
        
        print(f"      Fold {fold + 1} ✓", end=" ")
    print()

# ============== BLENDING + THRESHOLD OPTIMIZATION ==============
print("\n[6/6] Blending + Optimization du threshold...")

y_binary = (y > 0).astype(int)

# Test différents blends
best_acc = 0
best_blend = None
best_threshold = 0

for w_xgb in np.arange(0.3, 0.8, 0.1):
    w_cat = 1 - w_xgb
    blend_oof = w_xgb * oof_xgb + w_cat * oof_cat
    
    # Optimize threshold
    def neg_accuracy(threshold):
        pred = (blend_oof > threshold).astype(int)
        return -accuracy_score(y_binary, pred)
    
    result = minimize_scalar(neg_accuracy, bounds=(-0.1, 0.1), method='bounded')
    threshold = result.x
    acc = -result.fun
    
    if acc > best_acc:
        best_acc = acc
        best_blend = (w_xgb, w_cat)
        best_threshold = threshold

print(f"   Best blend: XGB={best_blend[0]:.1f}, CAT={best_blend[1]:.1f}")
print(f"   Optimal threshold: {best_threshold:.6f}")

# Final predictions
final_oof = best_blend[0] * oof_xgb + best_blend[1] * oof_cat
final_test = best_blend[0] * test_xgb + best_blend[1] * test_cat

# ============== RESULTS ==============
print("\n" + "=" * 70)
print("📊 RÉSULTATS V5 ULTIMATE")
print("=" * 70)

# Compare avec et sans threshold
acc_default = accuracy_score(y_binary, (final_oof > 0).astype(int))
acc_optimized = accuracy_score(y_binary, (final_oof > best_threshold).astype(int))

xgb_acc = accuracy_score(y_binary, (oof_xgb > 0).astype(int))
cat_acc = accuracy_score(y_binary, (oof_cat > 0).astype(int))

print(f"\n   XGBoost OOF:     {xgb_acc * 100:.2f}%")
print(f"   CatBoost OOF:    {cat_acc * 100:.2f}%")
print(f"   Blend (default): {acc_default * 100:.2f}%")
print(f"   Blend (optimized): {acc_optimized * 100:.2f}%")

print(f"\n   🏆 BEST OOF: {max(acc_default, acc_optimized) * 100:.2f}%")
print(f"   📈 Target: > 51.18% (previous best)")
print("=" * 70)

# Use threshold=0 for submission (more stable)
# Using optimized threshold might overfit
final_pred = (final_test > 0).astype(int)

# But test with threshold too
final_pred_thresh = (final_test > best_threshold).astype(int)

# Save both
submission = pd.DataFrame(final_pred, index=sample_submission.index, columns=['prediction'])
submission.to_csv(OUTPUT_PATH)

submission_thresh = pd.DataFrame(final_pred_thresh, index=sample_submission.index, columns=['prediction'])
submission_thresh.to_csv(OUTPUT_PATH.replace('.csv', '_threshold.csv'))

print(f"\n📁 Saved: {OUTPUT_PATH}")
print(f"📁 Saved: {OUTPUT_PATH.replace('.csv', '_threshold.csv')}")
print(f"📈 Predictions (default): {len(submission):,} - 1={final_pred.sum()} ({final_pred.mean()*100:.1f}%)")
print(f"📈 Predictions (threshold): 1={final_pred_thresh.sum()} ({final_pred_thresh.mean()*100:.1f}%)")

print("\n🔥 V5 ULTIMATE COMPLETE! 🔥")

gc.collect()
