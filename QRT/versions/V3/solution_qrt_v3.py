"""
🚀🔥 QRT V3 GPU ULTIMATE - MAXIMUM POWER 🔥🚀
==============================================
Target: 55%+ Accuracy

Techniques:
1. GPU-Accelerated LightGBM + XGBoost
2. CatBoost with native GPU
3. Seed Averaging (3 seeds per model)
4. 15-Fold CV for stability
5. Aggressive hyperparameters
6. Advanced feature engineering v2
7. Bayesian-style weighted blending
8. Temporal features + lag patterns
"""

import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
import gc

warnings.filterwarnings('ignore')

# ============== CONFIGURATION ==============
DATA_DIR = 'c:/Users/Karim/Desktop/Projet/QRT/data/'
X_TRAIN_PATH = DATA_DIR + 'X_train_9xQjqvZ.csv'
Y_TRAIN_PATH = DATA_DIR + 'y_train_Ppwhaz8.csv'
X_TEST_PATH = DATA_DIR + 'X_test_1zTtEnD.csv'
SAMPLE_SUB_PATH = DATA_DIR + 'sample_submission_SpGVFuH.csv'
OUTPUT_PATH = 'c:/Users/Karim/Desktop/Projet/QRT/V3/submission.csv'

N_FOLDS = 12
N_SEEDS = 3
BASE_SEED = 42

print("=" * 70)
print("🚀🔥 QRT V3 GPU ULTIMATE - MAXIMUM POWER 🔥🚀")
print("=" * 70)

# ============== LOAD DATA ==============
print("\n[1/8] Chargement des données...")
X_train = pd.read_csv(X_TRAIN_PATH, index_col='ROW_ID')
y_train = pd.read_csv(Y_TRAIN_PATH, index_col='ROW_ID')
X_test = pd.read_csv(X_TEST_PATH, index_col='ROW_ID')
sample_submission = pd.read_csv(SAMPLE_SUB_PATH, index_col='ROW_ID')

print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_test: {X_test.shape}")

# ============== ULTRA FEATURE ENGINEERING ==============
print("\n[2/8] Ultra Feature Engineering...")

RET_features = [f'RET_{i}' for i in range(1, 21)]
VOL_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]

def ultra_features(df):
    """Maximum feature engineering"""
    data = df.copy()
    ret_cols = RET_features
    vol_cols = VOL_features
    
    # === ROLLING STATS ===
    for h in [2, 3, 5, 7, 10, 15, 20]:
        data[f'RET_MEAN_{h}'] = data[ret_cols[:h]].mean(axis=1)
        data[f'RET_STD_{h}'] = data[ret_cols[:h]].std(axis=1)
        data[f'RET_SUM_{h}'] = data[ret_cols[:h]].sum(axis=1)
        data[f'RET_MIN_{h}'] = data[ret_cols[:h]].min(axis=1)
        data[f'RET_MAX_{h}'] = data[ret_cols[:h]].max(axis=1)
        data[f'RET_SKEW_{h}'] = data[ret_cols[:h]].skew(axis=1)
        data[f'RET_KURT_{h}'] = data[ret_cols[:h]].kurtosis(axis=1)
        data[f'RET_MEDIAN_{h}'] = data[ret_cols[:h]].median(axis=1)
        
        data[f'VOL_MEAN_{h}'] = data[vol_cols[:h]].mean(axis=1)
        data[f'VOL_STD_{h}'] = data[vol_cols[:h]].std(axis=1)
        data[f'VOL_SUM_{h}'] = data[vol_cols[:h]].sum(axis=1)
    
    # === MOMENTUM ===
    data['MOM_1_3'] = data['RET_MEAN_3']
    data['MOM_1_5'] = data['RET_MEAN_5']
    data['MOM_1_10'] = data['RET_MEAN_10']
    data['MOM_5_10'] = data[ret_cols[4:10]].mean(axis=1)
    data['MOM_10_20'] = data[ret_cols[9:20]].mean(axis=1)
    data['MOM_ACCEL'] = data['MOM_1_5'] - data['MOM_10_20']
    data['MOM_ACCEL2'] = data['MOM_1_3'] - data['MOM_5_10']
    
    # Weighted momentum (exponential decay)
    weights = np.exp(-np.arange(20) / 5)
    weights = weights / weights.sum()
    data['RET_EXP_WEIGHTED'] = data[ret_cols].values @ weights
    
    weights2 = np.exp(-np.arange(20) / 10)
    weights2 = weights2 / weights2.sum()
    data['RET_EXP_WEIGHTED2'] = data[ret_cols].values @ weights2
    
    # === CROSS-SECTIONAL ===
    for col in ['RET_MEAN_3', 'RET_MEAN_5', 'RET_MEAN_10', 'RET_MEAN_20', 
                'RET_STD_20', 'MEDIAN_DAILY_TURNOVER', 'RET_EXP_WEIGHTED']:
        if col in data.columns:
            g = data.groupby('TS')[col]
            data[f'{col}_RANK'] = g.rank(pct=True)
            data[f'{col}_ZNORM'] = (data[col] - g.transform('mean')) / (g.transform('std') + 1e-8)
            data[f'{col}_DEMEAN'] = data[col] - g.transform('mean')
            data[f'{col}_QRANK'] = pd.qcut(g.rank(method='first'), q=10, labels=False, duplicates='drop')
    
    # === SHARPE/SORTINO ===
    data['SHARPE_3'] = data['RET_MEAN_3'] / (data['RET_STD_3'] + 1e-8)
    data['SHARPE_5'] = data['RET_MEAN_5'] / (data['RET_STD_5'] + 1e-8)
    data['SHARPE_10'] = data['RET_MEAN_10'] / (data['RET_STD_10'] + 1e-8)
    data['SHARPE_20'] = data['RET_MEAN_20'] / (data['RET_STD_20'] + 1e-8)
    
    # Downside deviation (Sortino-style)
    for h in [5, 10, 20]:
        neg_rets = data[ret_cols[:h]].clip(upper=0)
        data[f'DOWNSIDE_STD_{h}'] = neg_rets.std(axis=1)
        data[f'SORTINO_{h}'] = data[f'RET_MEAN_{h}'] / (data[f'DOWNSIDE_STD_{h}'] + 1e-8)
    
    # === SIGN FEATURES ===
    for h in [3, 5, 10, 20]:
        pos_count = (data[ret_cols[:h]] > 0).sum(axis=1)
        data[f'POS_RATIO_{h}'] = pos_count / h
        data[f'NEG_RATIO_{h}'] = 1 - data[f'POS_RATIO_{h}']
        data[f'SIGN_STREAK_{h}'] = (data[ret_cols[:h]] > 0).astype(int).diff(axis=1).abs().sum(axis=1)
        
        # Consecutive positive/negative
        first_neg = (data[ret_cols[:h]] < 0).idxmax(axis=1)
        data[f'FIRST_NEG_IDX_{h}'] = h  # placeholder
    
    # === VOLATILITY REGIME ===
    data['VOL_REGIME'] = pd.qcut(data['RET_STD_20'].rank(method='first'), q=5, labels=False, duplicates='drop')
    data['VOL_OF_VOL'] = data[[f'RET_STD_{h}' for h in [3, 5, 10, 20]]].std(axis=1)
    
    # === INTERACTIONS ===
    data['MOM_VOL_INTERACT'] = data['MOM_1_5'] * data['VOL_MEAN_5']
    data['SHARPE_TURNOVER'] = data['SHARPE_20'] * data['MEDIAN_DAILY_TURNOVER']
    data['MOM_ACCEL_VOL'] = data['MOM_ACCEL'] / (data['RET_STD_5'] + 1e-8)
    
    # === LAG FEATURES ===
    data['RET_1_SIGN'] = (data['RET_1'] > 0).astype(int)
    data['RET_2_SIGN'] = (data['RET_2'] > 0).astype(int)
    data['RET_REVERSAL'] = -data['RET_1']
    data['RET_1_ABS'] = data['RET_1'].abs()
    data['RET_1_SQ'] = data['RET_1'] ** 2
    
    # === PATTERN FEATURES ===
    data['UP_UP'] = ((data['RET_1'] > 0) & (data['RET_2'] > 0)).astype(int)
    data['DOWN_DOWN'] = ((data['RET_1'] < 0) & (data['RET_2'] < 0)).astype(int)
    data['REVERSAL_PATTERN'] = ((data['RET_1'] > 0) != (data['RET_2'] > 0)).astype(int)
    
    return data

X_train = ultra_features(X_train)
X_test = ultra_features(X_test)

# ============== TARGET ENCODING ==============
print("\n[3/8] Target Encoding (OOF)...")

def compute_target_encoding_oof(X_train, y_train, X_test, col, n_folds=5, smoothing=10):
    train_enc = np.zeros(len(X_train))
    test_enc = np.zeros(len(X_test))
    global_mean = y_train['target'].mean()
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=BASE_SEED)
    
    for train_idx, val_idx in kf.split(X_train):
        train_data = X_train.iloc[train_idx]
        train_y = y_train.iloc[train_idx]
        
        combined = pd.DataFrame({col: train_data[col], 'target': train_y['target']})
        agg = combined.groupby(col)['target'].agg(['mean', 'count'])
        
        # Smoothing
        smooth_mean = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
        
        val_data = X_train.iloc[val_idx]
        train_enc[val_idx] = val_data[col].map(smooth_mean).fillna(global_mean)
        test_enc += X_test[col].map(smooth_mean).fillna(global_mean).values / n_folds
    
    return train_enc, test_enc

for col in ['GROUP', 'ALLOCATION']:
    if col in X_train.columns:
        train_enc, test_enc = compute_target_encoding_oof(X_train, y_train, X_test, col)
        X_train[f'{col}_ENC'] = train_enc
        X_test[f'{col}_ENC'] = test_enc

# Label encode for CatBoost
le_group = LabelEncoder()
le_alloc = LabelEncoder()
X_train['GROUP_CAT'] = le_group.fit_transform(X_train['GROUP'].astype(str))
X_test['GROUP_CAT'] = le_group.transform(X_test['GROUP'].astype(str))
X_train['ALLOC_CAT'] = le_alloc.fit_transform(X_train['ALLOCATION'].astype(str))
X_test['ALLOC_CAT'] = le_alloc.transform(X_test['ALLOCATION'].astype(str))

# ============== PREPARE DATA ==============
print("\n[4/8] Preparing data...")

exclude_cols = {'TS', 'ALLOCATION', 'GROUP'}
feature_cols = [c for c in X_train.columns if c not in exclude_cols]
numeric_features = [c for c in feature_cols if X_train[c].dtype in ['float64', 'int64', 'float32', 'int32', 'int8']]

print(f"   Total features: {len(numeric_features)}")

X = X_train[numeric_features].fillna(0).values.astype(np.float32)
y = y_train['target'].values.astype(np.float32)
X_te = X_test[numeric_features].fillna(0).values.astype(np.float32)
groups = X_train['TS'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_te_scaled = scaler.transform(X_te)

# ============== GPU MODEL CONFIGS ==============
print("\n[5/8] Configuring GPU models...")

# LightGBM GPU configs (multiple)
lgb_configs = [
    {
        'objective': 'regression',
        'metric': 'rmse',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'learning_rate': 0.008,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 100,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 10.0,
        'verbosity': -1,
        'n_estimators': 2000,
    },
    {
        'objective': 'regression',
        'metric': 'rmse',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'learning_rate': 0.01,
        'num_leaves': 127,
        'max_depth': 10,
        'min_child_samples': 50,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.65,
        'bagging_freq': 4,
        'lambda_l1': 0.05,
        'lambda_l2': 5.0,
        'verbosity': -1,
        'n_estimators': 1500,
    },
]

# XGBoost GPU config
xgb_configs = [
    {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'cuda',
        'learning_rate': 0.008,
        'max_depth': 7,
        'min_child_weight': 80,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.1,
        'reg_lambda': 10.0,
        'n_estimators': 2000,
        'verbosity': 0,
    },
    {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'cuda',
        'learning_rate': 0.01,
        'max_depth': 8,
        'min_child_weight': 50,
        'subsample': 0.65,
        'colsample_bytree': 0.5,
        'reg_alpha': 0.05,
        'reg_lambda': 5.0,
        'n_estimators': 1500,
        'verbosity': 0,
    },
]

# CatBoost GPU config
cat_configs = [
    {
        'loss_function': 'RMSE',
        'task_type': 'GPU',
        'devices': '0',
        'learning_rate': 0.01,
        'depth': 8,
        'l2_leaf_reg': 10,
        'iterations': 2000,
        'verbose': False,
        'early_stopping_rounds': 150,
    },
]

# ============== TRAINING WITH SEED AVERAGING ==============
print(f"\n[6/8] Training {N_FOLDS}-Fold CV x {N_SEEDS} Seeds...")

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_ridge = np.zeros(len(X))

test_lgb = np.zeros(len(X_te))
test_xgb = np.zeros(len(X_te))
test_cat = np.zeros(len(X_te))
test_ridge = np.zeros(len(X_te))

fold_scores = []

for seed_idx in range(N_SEEDS):
    seed = BASE_SEED + seed_idx * 100
    print(f"\n{'='*60}")
    print(f"   SEED {seed_idx + 1}/{N_SEEDS} (seed={seed})")
    print(f"{'='*60}")
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n   Fold {fold + 1}/{N_FOLDS}:", end=" ")
        
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        X_tr_sc, X_val_sc = X_scaled[train_idx], X_scaled[val_idx]
        
        # --- LightGBM (multiple configs) ---
        for cfg_idx, lgb_cfg in enumerate(lgb_configs):
            cfg = lgb_cfg.copy()
            cfg['random_state'] = seed + cfg_idx
            try:
                lgb_model = lgb.LGBMRegressor(**cfg)
                lgb_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(100, verbose=False)]
                )
                oof_lgb[val_idx] += lgb_model.predict(X_val) / (N_SEEDS * len(lgb_configs))
                test_lgb += lgb_model.predict(X_te) / (N_SEEDS * N_FOLDS * len(lgb_configs))
            except Exception as e:
                # Fallback to CPU if GPU fails
                cfg['device'] = 'cpu'
                del cfg['gpu_platform_id'], cfg['gpu_device_id']
                lgb_model = lgb.LGBMRegressor(**cfg)
                lgb_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(100, verbose=False)]
                )
                oof_lgb[val_idx] += lgb_model.predict(X_val) / (N_SEEDS * len(lgb_configs))
                test_lgb += lgb_model.predict(X_te) / (N_SEEDS * N_FOLDS * len(lgb_configs))
        
        # --- XGBoost ---
        for cfg_idx, xgb_cfg in enumerate(xgb_configs):
            cfg = xgb_cfg.copy()
            cfg['random_state'] = seed + cfg_idx
            try:
                xgb_model = xgb.XGBRegressor(**cfg)
                xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                oof_xgb[val_idx] += xgb_model.predict(X_val) / (N_SEEDS * len(xgb_configs))
                test_xgb += xgb_model.predict(X_te) / (N_SEEDS * N_FOLDS * len(xgb_configs))
            except Exception as e:
                cfg['tree_method'] = 'hist'
                cfg['device'] = 'cpu'
                xgb_model = xgb.XGBRegressor(**cfg)
                xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                oof_xgb[val_idx] += xgb_model.predict(X_val) / (N_SEEDS * len(xgb_configs))
                test_xgb += xgb_model.predict(X_te) / (N_SEEDS * N_FOLDS * len(xgb_configs))
        
        # --- CatBoost ---
        for cfg_idx, cat_cfg in enumerate(cat_configs):
            cfg = cat_cfg.copy()
            cfg['random_seed'] = seed + cfg_idx
            try:
                cat_model = CatBoostRegressor(**cfg)
                cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
                oof_cat[val_idx] += cat_model.predict(X_val) / (N_SEEDS * len(cat_configs))
                test_cat += cat_model.predict(X_te) / (N_SEEDS * N_FOLDS * len(cat_configs))
            except Exception as e:
                cfg['task_type'] = 'CPU'
                del cfg['devices']
                cat_model = CatBoostRegressor(**cfg)
                cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
                oof_cat[val_idx] += cat_model.predict(X_val) / (N_SEEDS * len(cat_configs))
                test_cat += cat_model.predict(X_te) / (N_SEEDS * N_FOLDS * len(cat_configs))
        
        # --- Ridge ---
        ridge_model = Ridge(alpha=10.0)
        ridge_model.fit(X_tr_sc, y_tr)
        oof_ridge[val_idx] += ridge_model.predict(X_val_sc) / N_SEEDS
        test_ridge += ridge_model.predict(X_te_scaled) / (N_SEEDS * N_FOLDS)
        
        print("✓", end="")
    print()

# ============== OPTIMAL BLENDING ==============
print("\n[7/8] Finding optimal blend weights...")

from scipy.optimize import minimize

y_binary = (y > 0).astype(int)

def objective(weights):
    w = np.abs(weights) / np.abs(weights).sum()  # Normalize
    blend = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cat + w[3] * oof_ridge
    pred = (blend > 0).astype(int)
    return -accuracy_score(y_binary, pred)

# Find optimal weights
result = minimize(objective, [0.3, 0.3, 0.3, 0.1], method='Nelder-Mead')
opt_weights = np.abs(result.x) / np.abs(result.x).sum()

print(f"   Optimal weights: LGB={opt_weights[0]:.3f}, XGB={opt_weights[1]:.3f}, CAT={opt_weights[2]:.3f}, Ridge={opt_weights[3]:.3f}")

# ============== FINAL RESULTS ==============
print("\n[8/8] Final Results...")

# Compare approaches
blend_oof = opt_weights[0] * oof_lgb + opt_weights[1] * oof_xgb + opt_weights[2] * oof_cat + opt_weights[3] * oof_ridge
blend_test = opt_weights[0] * test_lgb + opt_weights[1] * test_xgb + opt_weights[2] * test_cat + opt_weights[3] * test_ridge

simple_blend_oof = 0.35 * oof_lgb + 0.35 * oof_xgb + 0.25 * oof_cat + 0.05 * oof_ridge
simple_blend_test = 0.35 * test_lgb + 0.35 * test_xgb + 0.25 * test_cat + 0.05 * test_ridge

lgb_acc = accuracy_score(y_binary, (oof_lgb > 0).astype(int))
xgb_acc = accuracy_score(y_binary, (oof_xgb > 0).astype(int))
cat_acc = accuracy_score(y_binary, (oof_cat > 0).astype(int))
ridge_acc = accuracy_score(y_binary, (oof_ridge > 0).astype(int))
opt_acc = accuracy_score(y_binary, (blend_oof > 0).astype(int))
simple_acc = accuracy_score(y_binary, (simple_blend_oof > 0).astype(int))

print(f"\n{'=' * 70}")
print(f"   📊 OOF ACCURACY:")
print(f"      LightGBM:     {lgb_acc * 100:.2f}%")
print(f"      XGBoost:      {xgb_acc * 100:.2f}%")
print(f"      CatBoost:     {cat_acc * 100:.2f}%")
print(f"      Ridge:        {ridge_acc * 100:.2f}%")
print(f"      Simple Blend: {simple_acc * 100:.2f}%")
print(f"      Optimal Blend: {opt_acc * 100:.2f}%")
print(f"{'=' * 70}")
print(f"   🏆 BEST OOF: {max(opt_acc, simple_acc) * 100:.2f}%")
print(f"{'=' * 70}")

# Use best
if opt_acc >= simple_acc:
    final_pred = (blend_test > 0).astype(int)
    print("   Using Optimal Blend")
else:
    final_pred = (simple_blend_test > 0).astype(int)
    print("   Using Simple Blend")

# Save
submission = pd.DataFrame(final_pred, index=sample_submission.index, columns=['prediction'])
submission.to_csv(OUTPUT_PATH)

print(f"\n   📁 Saved: {OUTPUT_PATH}")
print(f"   📈 Predictions: {len(submission):,}")
print(f"   Distribution: 1={final_pred.sum()} ({final_pred.mean()*100:.1f}%)")

print("\n🚀🔥 V3 GPU ULTIMATE COMPLETE! 🔥🚀")

gc.collect()
