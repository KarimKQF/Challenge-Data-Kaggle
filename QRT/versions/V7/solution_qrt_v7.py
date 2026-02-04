"""
🚀 QRT V7 - AGGRESSIVE TARGET 52% 🚀
======================================
Strategy:
1. Enhanced Features: GROUP stats, Refined Momentum, Skeen/Kurt, CS-Ranking
2. Diverse Ensemble: standard LGB, DART, GOSS + Ridge
3. Robustness: 10-Fold CV x 5 Seeds
"""

import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import gc

warnings.filterwarnings('ignore')

# ============== CONFIGURATION ==============
DATA_DIR = 'c:/Users/Karim/Desktop/PROJET LABS/QRT/data/'
X_TRAIN_PATH = DATA_DIR + 'X_train_9xQjqvZ.csv'
Y_TRAIN_PATH = DATA_DIR + 'y_train_Ppwhaz8.csv'
X_TEST_PATH = DATA_DIR + 'X_test_1zTtEnD.csv'
SAMPLE_SUB_PATH = DATA_DIR + 'sample_submission_SpGVFuH.csv'
OUTPUT_PATH = 'c:/Users/Karim/Desktop/PROJET LABS/QRT/V7/submission.csv'

N_FOLDS = 10
N_SEEDS = 5
BASE_SEED = 42

LGB_CONFIGS = [
    {   # 1. Standard Conservative
        'objective': 'mse', 'metric': 'mse', 'learning_rate': 0.015,
        'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 80,
        'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5,
        'verbosity': -1, 'n_estimators': 500,
    },
    {   # 2. DART (Généralisation)
        'objective': 'mse', 'metric': 'mse', 'learning_rate': 0.02,
        'boosting_type': 'dart', 'num_leaves': 10, 'max_depth': 3,
        'feature_fraction': 0.6, 'verbosity': -1, 'n_estimators': 400,
    },
    {   # 3. GOSS
        'objective': 'mse', 'metric': 'mse', 'learning_rate': 0.01,
        'boosting_type': 'goss', 'num_leaves': 20, 'max_depth': 5,
        'feature_fraction': 0.5, 'verbosity': -1, 'n_estimators': 600,
    }
]

print("=" * 70)
print("🚀 QRT V7 - TARGET 52% ACCURACY 🚀")
print("=" * 70)

# ============== LOAD DATA ==============
print("\n[1/5] Chargement des données...")
X_train = pd.read_csv(X_TRAIN_PATH, index_col='ROW_ID')
y_train = pd.read_csv(Y_TRAIN_PATH, index_col='ROW_ID')
X_test = pd.read_csv(X_TEST_PATH, index_col='ROW_ID')
sample_submission = pd.read_csv(SAMPLE_SUB_PATH, index_col='ROW_ID')

# ============== FEATURE ENGINEERING ==============
print("\n[2/5] Feature engineering avancé...")

def build_features_v7(df):
    data = df.copy()
    RET_features = [f'RET_{i}' for i in range(1, 21)]
    
    # 1. Base Rolling Performance
    for i in [3, 5, 10, 15, 20]:
        data[f'AVG_RET_{i}'] = data[RET_features[:i]].mean(axis=1)
        data[f'TS_AVG_RET_{i}'] = data.groupby('TS')[f'AVG_RET_{i}'].transform('mean')
    
    # 2. Group Performance (The Secret Sauce)
    for i in [5, 20]:
        data[f'GROUP_AVG_RET_{i}'] = data.groupby(['TS', 'GROUP'])[f'AVG_RET_{i}'].transform('mean')
        
    # 3. Volatility, Skew, Kurt
    data['RET_STD_20'] = data[RET_features].std(axis=1)
    data['RET_SKEW_20'] = data[RET_features].skew(axis=1)
    data['RET_KURT_20'] = data[RET_features].kurtosis(axis=1)
    
    # 4. Momentum Horizons
    data['MOM_3'] = data[RET_features[:3]].sum(axis=1)
    data['MOM_10'] = data[RET_features[:10]].sum(axis=1)
    data['MOM_20'] = data[RET_features[:20]].sum(axis=1)
    
    # 5. Cross-sectional Ranking
    data['RANK_RET_5'] = data.groupby('TS')['AVG_RET_5'].rank(pct=True)
    
    # 6. Volumes
    VOL_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
    data['VOL_AVG_20'] = data[VOL_features].mean(axis=1)
    
    return data

X_train = build_features_v7(X_train)
X_test = build_features_v7(X_test)

exclude = ['TS', 'GROUP', 'ALLOCATION']
features = [c for c in X_train.columns if c not in exclude]
print(f"   Nombre de features: {len(features)}")

# ============== TRAINING ==============
print(f"\n[3/5] Training {N_FOLDS}-Fold CV x {N_SEEDS} seeds...")

train_dates = X_train['TS'].unique()
oof_pred = np.zeros(len(X_train))
test_pred = np.zeros(len(X_test))
all_scores = []

# For Ridge
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_train[features].fillna(0))
X_te_scaled = scaler.transform(X_test[features].fillna(0))

for seed_idx in range(N_SEEDS):
    seed = BASE_SEED + seed_idx * 100
    print(f"\n   === SEED {seed_idx + 1}/{N_SEEDS} (seed={seed}) ===")
    
    kf = KFold(n_splits=N_FOLDS, random_state=seed, shuffle=True)
    
    for fold, (train_date_idx, val_date_idx) in enumerate(kf.split(train_dates)):
        local_train_dates = train_dates[train_date_idx]
        local_val_dates = train_dates[val_date_idx]
        
        train_mask = X_train['TS'].isin(local_train_dates)
        val_mask = X_train['TS'].isin(local_val_dates)
        
        # Data prepared
        X_tr, y_tr = X_train.loc[train_mask, features].fillna(0).values, y_train.loc[train_mask, 'target'].values
        X_val, y_val = X_train.loc[val_mask, features].fillna(0).values, y_train.loc[val_mask, 'target'].values
        
        # Ensemble predictions for this fold
        fold_val_pred = np.zeros(len(y_val))
        fold_test_pred = np.zeros(len(X_test))
        
        # 1. Diverse LightGBMs (90% Weight)
        for cfg in LGB_CONFIGS:
            m_lgb = lgb.LGBMRegressor(**cfg, random_state=seed)
            m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
            
            fold_val_pred += (0.9 / len(LGB_CONFIGS)) * m_lgb.predict(X_val)
            fold_test_pred += (0.9 / len(LGB_CONFIGS)) * m_lgb.predict(X_test[features].fillna(0).values)
            
        # 2. Ridge (10% Weight)
        X_tr_r = X_tr_scaled[train_mask]
        X_val_r = X_tr_scaled[val_mask]
        m_ridge = Ridge(alpha=10.0)
        m_ridge.fit(X_tr_r, y_tr)
        
        fold_val_pred += 0.1 * m_ridge.predict(X_val_r)
        fold_test_pred += 0.1 * m_ridge.predict(X_te_scaled)
        
        # Accumulate
        oof_pred[val_mask] += fold_val_pred / N_SEEDS
        test_pred += fold_test_pred / (N_FOLDS * N_SEEDS)
        
        score = accuracy_score((y_val > 0).astype(int), (fold_val_pred > 0).astype(int))
        all_scores.append(score)
        print(f"      Fold {fold+1}: {score*100:.2f}%", end=" ")
    print()

# ============== RESULTS ==============
print("\n[4/5] Résultats finaux...")
final_acc = accuracy_score((y_train['target'] > 0).astype(int), (oof_pred > 0).astype(int))
print(f"\n{'=' * 70}")
print(f"   🏆 OOF ACCURACY V7: {final_acc * 100:.2f}%")
print(f"   📊 Mean Fold Accuracy: {np.mean(all_scores) * 100:.2f}%")
print(f"   🎯 Target: 52.00%")
print(f"{'=' * 70}")

# ============== SAVE ==============
print("\n[5/5] Soumission...")
pd.DataFrame((test_pred > 0).astype(int), index=sample_submission.index, columns=['prediction']).to_csv(OUTPUT_PATH)
print(f"   Fichier: {OUTPUT_PATH}")
gc.collect()
