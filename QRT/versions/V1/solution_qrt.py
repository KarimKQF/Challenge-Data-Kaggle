"""
QRT Challenge Solution - Prédiction du signe des allocations d'actifs
======================================================================
Basé sur l'architecture V7 du projet Kaggle, adapté pour:
- Classification binaire (signe du rendement)
- Métrique: Accuracy
- Features: RET_*, SIGNED_VOLUME_*, MEDIAN_DAILY_TURNOVER, GROUP

Benchmark de référence: 0.5079 (public leaderboard)
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
DATA_DIR = 'c:/Users/Karim/Desktop/Projet/QRT/data/'
X_TRAIN_PATH = DATA_DIR + 'X_train_9xQjqvZ.csv'
Y_TRAIN_PATH = DATA_DIR + 'y_train_Ppwhaz8.csv'
X_TEST_PATH = DATA_DIR + 'X_test_1zTtEnD.csv'
SAMPLE_SUB_PATH = DATA_DIR + 'sample_submission_SpGVFuH.csv'
OUTPUT_PATH = 'c:/Users/Karim/Desktop/Projet/QRT/V1/submission.csv'

N_FOLDS = 8
RANDOM_SEED = 42

# Multiple diverse LightGBM configs for ensemble
LGB_CONFIGS = [
    {   # Config 1: Conservative (comme le benchmark)
        'objective': 'mse',
        'metric': 'mse',
        'learning_rate': 0.01,
        'num_leaves': 8,  # max_depth=3 equiv
        'max_depth': 3,
        'min_child_samples': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbosity': -1,
        'n_estimators': 500,
        'random_state': RANDOM_SEED,
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
        'random_state': RANDOM_SEED + 1,
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
        'random_state': RANDOM_SEED + 2,
    },
]

print("=" * 70)
print("QRT CHALLENGE SOLUTION - Prédiction du signe des allocations")
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

# ============== FEATURE ENGINEERING ==============
print("\n[2/5] Feature engineering...")

# Features de base
RET_features = [f'RET_{i}' for i in range(1, 21)]
SIGNED_VOLUME_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
TURNOVER_features = ['MEDIAN_DAILY_TURNOVER']

def build_features(df):
    """Construction des features additionnelles (comme le benchmark)"""
    data = df.copy()
    
    # 1. Moyennes des performances à différents horizons
    for i in [3, 5, 10, 15, 20]:
        data[f'AVERAGE_PERF_{i}'] = data[RET_features[:i]].mean(axis=1)
        data[f'ALLOCATIONS_AVERAGE_PERF_{i}'] = data.groupby('TS')[f'AVERAGE_PERF_{i}'].transform('mean')
    
    # 2. Volatilité des performances
    for i in [20]:
        data[f'STD_PERF_{i}'] = data[RET_features[:i]].std(axis=1)
        data[f'ALLOCATIONS_STD_PERF_{i}'] = data.groupby('TS')[f'STD_PERF_{i}'].transform('mean')
    
    # 3. Features de momentum
    data['MOMENTUM_SHORT'] = data[RET_features[:5]].sum(axis=1)
    data['MOMENTUM_LONG'] = data[RET_features[:20]].sum(axis=1)
    data['MOMENTUM_DIFF'] = data['MOMENTUM_SHORT'] - data['MOMENTUM_LONG'] / 4
    
    # 4. Features de volume
    data['VOLUME_SHORT'] = data[SIGNED_VOLUME_features[:5]].mean(axis=1)
    data['VOLUME_LONG'] = data[SIGNED_VOLUME_features[:20]].mean(axis=1)
    
    # 5. Ratios
    data['RET_VOL_RATIO'] = data['AVERAGE_PERF_20'] / (data['STD_PERF_20'] + 1e-8)
    
    return data

X_train = build_features(X_train)
X_test = build_features(X_test)

# Liste des features pour le modèle
features = RET_features + SIGNED_VOLUME_features + TURNOVER_features
features = features + [f'AVERAGE_PERF_{i}' for i in [3, 5, 10, 15, 20]]
features = features + [f'ALLOCATIONS_AVERAGE_PERF_{i}' for i in [3, 5, 10, 15, 20]]
features = features + [f'STD_PERF_{i}' for i in [20]]
features = features + [f'ALLOCATIONS_STD_PERF_{i}' for i in [20]]
features = features + ['MOMENTUM_SHORT', 'MOMENTUM_LONG', 'MOMENTUM_DIFF']
features = features + ['VOLUME_SHORT', 'VOLUME_LONG', 'RET_VOL_RATIO']

print(f"   Nombre de features: {len(features)}")

# ============== K-FOLD CROSS-VALIDATION ==============
print(f"\n[3/5] Training avec {N_FOLDS}-Fold CV + {len(LGB_CONFIGS)} configs ensemble...")

# K-Fold sur les dates pour éviter data leakage
train_dates = X_train['TS'].unique()
kf = KFold(n_splits=N_FOLDS, random_state=RANDOM_SEED, shuffle=True)

oof_pred = np.zeros(len(X_train))
test_pred = np.zeros(len(X_test))
fold_scores = []

for fold, (train_date_idx, val_date_idx) in enumerate(kf.split(train_dates)):
    print(f"\n   Fold {fold + 1}/{N_FOLDS}:")
    
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
    
    # Ensemble de configs
    for cfg_idx, lgb_cfg in enumerate(LGB_CONFIGS):
        model = lgb.LGBMRegressor(**lgb_cfg)
        
        model.fit(
            X_tr.values, y_tr.values,
            eval_set=[(X_val.values, y_val.values)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        fold_val_pred += model.predict(X_val.values) / len(LGB_CONFIGS)
        fold_test_pred += model.predict(X_test[features].fillna(0).values) / len(LGB_CONFIGS)
    
    # Store OOF predictions
    oof_pred[val_mask] = fold_val_pred
    test_pred += fold_test_pred / N_FOLDS
    
    # Score du fold (accuracy sur le signe)
    fold_accuracy = accuracy_score((y_val > 0).astype(int), (fold_val_pred > 0).astype(int))
    fold_scores.append(fold_accuracy)
    print(f"      Accuracy: {fold_accuracy * 100:.2f}%")

# ============== FINAL RESULTS ==============
print("\n[4/5] Résultats finaux...")

# OOF accuracy
final_accuracy = accuracy_score(
    (y_train['target'] > 0).astype(int),
    (oof_pred > 0).astype(int)
)

mean_fold = np.mean(fold_scores)
std_fold = np.std(fold_scores)

print(f"\n{'=' * 70}")
print(f"   🏆 OOF ACCURACY: {final_accuracy * 100:.2f}%")
print(f"   📊 Mean Fold Accuracy: {mean_fold * 100:.2f}% (+/- {std_fold * 100:.2f}%)")
print(f"   📈 Benchmark: 50.79% (public leaderboard)")
print(f"{'=' * 70}")

# ============== SAVE SUBMISSION ==============
print("\n[5/5] Génération du fichier de soumission...")

# Convert predictions to binary (0 or 1)
predictions = (test_pred > 0).astype(int)

submission = pd.DataFrame(predictions, index=sample_submission.index, columns=['prediction'])
submission.to_csv(OUTPUT_PATH)

print(f"   Fichier sauvegardé: {OUTPUT_PATH}")
print(f"   Nombre de prédictions: {len(submission):,}")
print(f"   Distribution: 1={predictions.sum()} ({predictions.mean()*100:.1f}%), 0={len(predictions)-predictions.sum()} ({(1-predictions.mean())*100:.1f}%)")

print("\n🚀 SOLUTION COMPLETE! 🚀")

# Cleanup
gc.collect()
