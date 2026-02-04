"""
🎯 QRT V4 ANTI-OVERFITTING SOLUTION 🎯
=======================================
Goal: Minimize OOF-to-Leaderboard gap

Strategy:
- Minimal features (benchmark-style)
- Ultra-regularized LightGBM (depth=3)
- 8-fold GroupKFold on dates
- Single model, no OOF optimization
- Pessimistic validation
"""

import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import gc

warnings.filterwarnings('ignore')

# ============== CONFIGURATION ==============
DATA_DIR = 'c:/Users/Karim/Desktop/PROJET LABS/QRT/data/'
X_TRAIN_PATH = DATA_DIR + 'X_train_9xQjqvZ.csv'
Y_TRAIN_PATH = DATA_DIR + 'y_train_Ppwhaz8.csv'
X_TEST_PATH = DATA_DIR + 'X_test_1zTtEnD.csv'
SAMPLE_SUB_PATH = DATA_DIR + 'sample_submission_SpGVFuH.csv'
OUTPUT_PATH = 'c:/Users/Karim/Desktop/PROJET LABS/QRT/V4/submission.csv'

N_FOLDS = 8
SEED = 42

print("=" * 70)
print("🎯 QRT V4 ANTI-OVERFITTING SOLUTION 🎯")
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

# ============== MINIMAL FEATURE ENGINEERING ==============
print("\n[2/5] Feature Engineering MINIMAL (anti-overfitting)...")

RET_features = [f'RET_{i}' for i in range(1, 21)]
VOL_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]

def minimal_features(df):
    """
    Features MINIMALES - exactement comme le benchmark officiel
    Objectif: PAS d'overfitting
    """
    data = df.copy()
    
    # === ROLLING MEANS (benchmark exact) ===
    for i in [3, 5, 10, 15, 20]:
        data[f'AVERAGE_PERF_{i}'] = data[RET_features[:i]].mean(axis=1)
        # Cross-sectional mean per timestamp
        data[f'ALLOCATIONS_AVERAGE_PERF_{i}'] = data.groupby('TS')[f'AVERAGE_PERF_{i}'].transform('mean')
    
    # === ROLLING STD (benchmark exact - only 20) ===
    for i in [20]:
        data[f'STD_PERF_{i}'] = data[RET_features[:i]].std(axis=1)
        data[f'ALLOCATIONS_STD_PERF_{i}'] = data.groupby('TS')[f'STD_PERF_{i}'].transform('mean')
    
    # === VOLUME FEATURES (simple) ===
    for i in [5, 10, 20]:
        data[f'VOL_MEAN_{i}'] = data[VOL_features[:i]].mean(axis=1)
    
    return data

X_train = minimal_features(X_train)
X_test = minimal_features(X_test)

# ============== PREPARE FEATURES ==============
print("\n[3/5] Préparation des features...")

# Feature list exactement comme benchmark
features = RET_features + VOL_features + ['MEDIAN_DAILY_TURNOVER']
features = features + [f'AVERAGE_PERF_{i}' for i in [3, 5, 10, 15, 20]]
features = features + [f'ALLOCATIONS_AVERAGE_PERF_{i}' for i in [3, 5, 10, 15, 20]]
features = features + [f'STD_PERF_{20}']
features = features + [f'ALLOCATIONS_STD_PERF_{20}']
features = features + [f'VOL_MEAN_{i}' for i in [5, 10, 20]]

print(f"   Total features: {len(features)}")
print(f"   Features: {features[:10]}...")

X = X_train[features].fillna(0).values.astype(np.float32)
y = y_train['target'].values.astype(np.float32)
X_te = X_test[features].fillna(0).values.astype(np.float32)

# Get unique dates for GroupKFold
train_dates = X_train['TS'].unique()

# ============== MODEL CONFIG (ULTRA-REGULARIZED) ==============
print("\n[4/5] Configuration modèle ULTRA-RÉGULARISÉ...")

# Paramètres IDENTIQUES au benchmark officiel
lgb_params = {
    'objective': 'mse',
    'metric': 'mse',
    'num_threads': 8,
    'seed': SEED,
    'verbosity': -1,
    'learning_rate': 0.01,  # Benchmark: 1e-2
    'max_depth': 3,          # Benchmark: 3 (très conservateur)
}
NUM_BOOST_ROUND = 500  # Benchmark: 500

print(f"   learning_rate: {lgb_params['learning_rate']}")
print(f"   max_depth: {lgb_params['max_depth']}")
print(f"   num_boost_round: {NUM_BOOST_ROUND}")

# ============== TRAINING (BENCHMARK STYLE) ==============
print(f"\n[5/5] Training {N_FOLDS}-Fold CV (dates, not samples)...")

scores = []
models = []
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_te))

# KFold on DATES (exactly like benchmark)
splits = KFold(n_splits=N_FOLDS, random_state=0, shuffle=True).split(train_dates)

for fold, (local_train_dates_ids, local_test_dates_ids) in enumerate(splits):
    local_train_dates = train_dates[local_train_dates_ids]
    local_test_dates = train_dates[local_test_dates_ids]
    
    local_train_ids = X_train['TS'].isin(local_train_dates)
    local_test_ids = X_train['TS'].isin(local_test_dates)
    
    X_tr = X_train.loc[local_train_ids, features].fillna(0)
    y_tr = y_train.loc[local_train_ids, 'target']
    
    X_val = X_train.loc[local_test_ids, features].fillna(0)
    y_val = y_train.loc[local_test_ids, 'target']
    
    # Train LightGBM
    train_data = lgb.Dataset(X_tr, label=y_tr.values)
    model = lgb.train(lgb_params, train_data, num_boost_round=NUM_BOOST_ROUND)
    models.append(model)
    
    # OOF predictions
    val_pred = model.predict(X_val.values)
    oof_preds[local_test_ids.values] = val_pred
    
    # Test predictions (average across folds)
    test_preds += model.predict(X_test[features].fillna(0).values) / N_FOLDS
    
    # Score
    acc = accuracy_score((y_val > 0).astype(int), (val_pred > 0).astype(int))
    scores.append(acc)
    print(f"   Fold {fold + 1} - Accuracy: {acc * 100:.2f}%")

# ============== RESULTS ==============
print("\n" + "=" * 70)
print("📊 RÉSULTATS V4 ANTI-OVERFITTING")
print("=" * 70)

mean_score = np.mean(scores)
std_score = np.std(scores)
min_score = np.min(scores)
max_score = np.max(scores)

print(f"\n   Mean Accuracy: {mean_score * 100:.2f}%")
print(f"   Std:           ±{std_score * 100:.2f}%")
print(f"   Range:         [{min_score * 100:.2f}% - {max_score * 100:.2f}%]")

# OOF global accuracy
y_binary = (y_train['target'] > 0).astype(int)
oof_binary = (oof_preds > 0).astype(int)
oof_accuracy = accuracy_score(y_binary, oof_binary)

print(f"\n   OOF Global Accuracy: {oof_accuracy * 100:.2f}%")

# Pessimistic estimate (worst fold)
print(f"\n   ⚠️  Estimation PESSIMISTE (pire fold): {min_score * 100:.2f}%")

# Overfitting warning
if oof_accuracy > 0.53:
    print("\n   ⚠️  ATTENTION: OOF > 53% - risque d'overfitting!")
else:
    print("\n   ✅ OOF raisonnable - bon signe de généralisation")

# ============== SAVE SUBMISSION ==============
print("\n" + "=" * 70)

final_pred = (test_preds > 0).astype(int)
submission = pd.DataFrame(final_pred, index=sample_submission.index, columns=['prediction'])
submission.to_csv(OUTPUT_PATH)

print(f"📁 Saved: {OUTPUT_PATH}")
print(f"📈 Predictions: {len(submission):,}")
print(f"   Distribution: 1={final_pred.sum()} ({final_pred.mean()*100:.1f}%)")

print("\n" + "=" * 70)
print("🎯 V4 ANTI-OVERFITTING COMPLETE!")
print("   Expected LB Score: ~{:.1f}% (close to OOF)".format(min_score * 100))
print("=" * 70)

gc.collect()
