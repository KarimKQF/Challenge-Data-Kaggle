"""
🏆 QRT V6 - RETOUR AUX SOURCES + MICRO-OPTIMISATIONS 🏆
========================================================
Constat: V1 (LightGBM Ensemble 8-Fold, 51.18%) = MEILLEUR SCORE
        Plus on complexifie = plus ça baisse

Stratégie V6:
1. EXACTEMENT la même base que V1
2. Plus de seeds (5 au lieu de 1)
3. Légères variations des hyperparamètres
4. Aucune feature supplémentaire complexe
5. Objectif: 51.2%+
"""

import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
import gc
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# ============== CONFIGURATION ==============
DATA_DIR = 'c:/Users/Karim/Desktop/PROJET LABS/QRT/data/'
X_TRAIN_PATH = DATA_DIR + 'X_train_9xQjqvZ.csv'
Y_TRAIN_PATH = DATA_DIR + 'y_train_Ppwhaz8.csv'
X_TEST_PATH = DATA_DIR + 'X_test_1zTtEnD.csv'
SAMPLE_SUB_PATH = DATA_DIR + 'sample_submission_SpGVFuH.csv'
OUTPUT_PATH = 'c:/Users/Karim/Desktop/PROJET LABS/QRT/V6/submission.csv'

N_FOLDS = 8  # Comme V1
N_SEEDS = 5  # Plus de seeds pour stabilité
BASE_SEED = 42

# EXACTEMENT les configs de V1 qui ont marché
LGB_CONFIGS = [
    {   # Config 1: Conservative (benchmark)
        'objective': 'mse',
        'metric': 'mse',
        'learning_rate': 0.01,
        'num_leaves': 8,
        'max_depth': 3,
        'min_child_samples': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbosity': -1,
        'n_estimators': 500,
    },
    {   # Config 2: Moderate
        'objective': 'mse',
        'metric': 'mse',
        'learning_rate': 0.015,
        'num_leaves': 15,
        'max_depth': 4,
        'min_child_samples': 80,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.65,
        'bagging_freq': 4,
        'lambda_l1': 0.05,
        'lambda_l2': 0.8,
        'verbosity': -1,
        'n_estimators': 400,
    },
    {   # Config 3: Slightly more complex
        'objective': 'mse',
        'metric': 'mse',
        'learning_rate': 0.02,
        'num_leaves': 20,
        'max_depth': 5,
        'min_child_samples': 60,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.6,
        'bagging_freq': 3,
        'lambda_l1': 0.02,
        'lambda_l2': 0.5,
        'verbosity': -1,
        'n_estimators': 300,
    },
]

print("=" * 70)
print("🏆 QRT V6 - RETOUR AUX SOURCES (V1) + SEED AVERAGING 🏆")
print("=" * 70)

# ============== LOAD DATA ==============
print("\n[1/5] Chargement des données...")
X_train = pd.read_csv(X_TRAIN_PATH, index_col='ROW_ID')
y_train = pd.read_csv(Y_TRAIN_PATH, index_col='ROW_ID')
X_test = pd.read_csv(X_TEST_PATH, index_col='ROW_ID')
sample_submission = pd.read_csv(SAMPLE_SUB_PATH, index_col='ROW_ID')

print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_test: {X_test.shape}")

# ============== FEATURE ENGINEERING (EXACTEMENT V1) ==============
print("\n[2/5] Feature engineering (EXACTEMENT comme V1)...")

RET_features = [f'RET_{i}' for i in range(1, 21)]
SIGNED_VOLUME_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
TURNOVER_features = ['MEDIAN_DAILY_TURNOVER']

def build_features(df):
    """Construction des features (IDENTIQUE à V1)"""
    data = df.copy()
    
    # 1. Moyennes des performances
    for i in [3, 5, 10, 15, 20]:
        data[f'AVERAGE_PERF_{i}'] = data[RET_features[:i]].mean(axis=1)
        data[f'ALLOCATIONS_AVERAGE_PERF_{i}'] = data.groupby('TS')[f'AVERAGE_PERF_{i}'].transform('mean')
    
    # 2. Volatilité
    for i in [20]:
        data[f'STD_PERF_{i}'] = data[RET_features[:i]].std(axis=1)
        data[f'ALLOCATIONS_STD_PERF_{i}'] = data.groupby('TS')[f'STD_PERF_{i}'].transform('mean')
    
    # 3. Momentum
    data['MOMENTUM_SHORT'] = data[RET_features[:5]].sum(axis=1)
    data['MOMENTUM_LONG'] = data[RET_features[:20]].sum(axis=1)
    data['MOMENTUM_DIFF'] = data['MOMENTUM_SHORT'] - data['MOMENTUM_LONG'] / 4
    
    # 4. Volume
    data['VOLUME_SHORT'] = data[SIGNED_VOLUME_features[:5]].mean(axis=1)
    data['VOLUME_LONG'] = data[SIGNED_VOLUME_features[:20]].mean(axis=1)
    
    # 5. Ratio
    data['RET_VOL_RATIO'] = data['AVERAGE_PERF_20'] / (data['STD_PERF_20'] + 1e-8)
    
    return data

X_train = build_features(X_train)
X_test = build_features(X_test)

# Features (IDENTIQUE à V1)
features = RET_features + SIGNED_VOLUME_features + TURNOVER_features
features = features + [f'AVERAGE_PERF_{i}' for i in [3, 5, 10, 15, 20]]
features = features + [f'ALLOCATIONS_AVERAGE_PERF_{i}' for i in [3, 5, 10, 15, 20]]
features = features + [f'STD_PERF_{i}' for i in [20]]
features = features + [f'ALLOCATIONS_STD_PERF_{i}' for i in [20]]
features = features + ['MOMENTUM_SHORT', 'MOMENTUM_LONG', 'MOMENTUM_DIFF']
features = features + ['VOLUME_SHORT', 'VOLUME_LONG', 'RET_VOL_RATIO']

print(f"   Nombre de features: {len(features)}")

# ============== TRAINING AVEC SEED AVERAGING ==============
print(f"\n[3/5] Training {N_FOLDS}-Fold CV x {len(LGB_CONFIGS)} configs x {N_SEEDS} seeds...")

train_dates = X_train['TS'].unique()

oof_pred = np.zeros(len(X_train))
test_pred = np.zeros(len(X_test))
all_fold_scores = []

for seed_idx in range(N_SEEDS):
    seed = BASE_SEED + seed_idx * 100
    print(f"\n   === SEED {seed_idx + 1}/{N_SEEDS} (seed={seed}) ===")
    
    kf = KFold(n_splits=N_FOLDS, random_state=seed, shuffle=True)
    
    for fold, (train_date_idx, val_date_idx) in enumerate(kf.split(train_dates)):
        local_train_dates = train_dates[train_date_idx]
        local_val_dates = train_dates[val_date_idx]
        
        train_mask = X_train['TS'].isin(local_train_dates)
        val_mask = X_train['TS'].isin(local_val_dates)
        
        X_tr = X_train.loc[train_mask, features].fillna(0)
        y_tr = y_train.loc[train_mask, 'target']
        
        X_val = X_train.loc[val_mask, features].fillna(0)
        y_val = y_train.loc[val_mask, 'target']
        
        fold_val_pred = np.zeros(len(X_val))
        fold_test_pred = np.zeros(len(X_test))
        
        # Ensemble des 3 configs
        for cfg_idx, lgb_cfg in enumerate(LGB_CONFIGS):
            cfg = lgb_cfg.copy()
            cfg['random_state'] = seed + cfg_idx
            
            model = lgb.LGBMRegressor(**cfg)
            
            model.fit(
                X_tr.values, y_tr.values,
                eval_set=[(X_val.values, y_val.values)],
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
            
            fold_val_pred += model.predict(X_val.values) / len(LGB_CONFIGS)
            fold_test_pred += model.predict(X_test[features].fillna(0).values) / len(LGB_CONFIGS)
        
        # Accumulate avec seed averaging
        oof_pred[val_mask] += fold_val_pred / N_SEEDS
        test_pred += fold_test_pred / (N_FOLDS * N_SEEDS)
        
        fold_accuracy = accuracy_score((y_val > 0).astype(int), (fold_val_pred > 0).astype(int))
        all_fold_scores.append(fold_accuracy)
        print(f"      Fold {fold + 1}: {fold_accuracy * 100:.2f}%", end=" ")
    print()

# ============== RESULTS ==============
print("\n[4/5] Résultats finaux...")

final_accuracy = accuracy_score(
    (y_train['target'] > 0).astype(int),
    (oof_pred > 0).astype(int)
)

mean_fold = np.mean(all_fold_scores)
std_fold = np.std(all_fold_scores)

print(f"\n{'=' * 70}")
print(f"   🏆 OOF ACCURACY: {final_accuracy * 100:.2f}%")
print(f"   📊 Mean Fold Accuracy: {mean_fold * 100:.2f}% (+/- {std_fold * 100:.2f}%)")
print(f"   📈 Target: > 51.18% (V1 best)")
print(f"{'=' * 70}")

# ============== SAVE ==============
print("\n[5/5] Génération du fichier de soumission...")

predictions = (test_pred > 0).astype(int)
submission = pd.DataFrame(predictions, index=sample_submission.index, columns=['prediction'])
submission.to_csv(OUTPUT_PATH)

print(f"   Fichier: {OUTPUT_PATH}")
print(f"   Predictions: {len(submission):,}")
print(f"   Distribution: 1={predictions.sum()} ({predictions.mean()*100:.1f}%)")

print("\n🏆 V6 COMPLETE! 🏆")

gc.collect()
