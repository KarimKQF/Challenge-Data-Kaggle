
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import warnings
import lightgbm as lgb
import gc

warnings.filterwarnings('ignore')

# Define paths
# train_path = '/kaggle/input/ts-forecasting/train.parquet'
# test_path = '/kaggle/input/ts-forecasting/test.parquet'
# Adjusted for local environment
train_path = 'c:/Users/Karim/Desktop/Kaggle/data/train.parquet'
test_path = 'c:/Users/Karim/Desktop/Kaggle/data/test.parquet'

val_threshold = 3500

print("Computing Stats...")
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

def build_context_features(data, enc_stats=None):
    x = data.copy()

    # Encoded categorical signals
    if enc_stats is not None:
        for c in ['sub_category', 'sub_code']:
            x[c + '_enc'] = x[c].map(enc_stats[c]).fillna(enc_stats['global_mean'])

    # Interaction features
    x['d_al_am'] = x['feature_al'] - x['feature_am']
    x['r_al_am'] = x['feature_al'] / (x['feature_am'] + 1e-7)
    x['d_cg_by'] = x['feature_cg'] - x['feature_by']

    # Cross-sectional normalization
    norm_cols = ['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'd_al_am']

    for col in norm_cols:
        g = x.groupby('ts_index')[col]
        x[col + '_cs'] = (x[col] - g.transform('mean')) / (g.transform('std') + 1e-7)

    # Temporal signal
    x['t_cycle'] = np.sin(2 * np.pi * x['ts_index'] / 100)

    return x

forecast_windows = [1, 3, 10, 25]
test_outputs = []
cv_cache = {'y': [], 'pred': [], 'wt': []}

lgb_cfg = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.015,
    'n_estimators': 4000,
    'num_leaves': 80,
    'min_child_samples': 200,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 10.0,
    'verbosity': -1,
    'device': 'gpu' # Trying to use GPU if available, as requested in previous tasks
}

# Fallback to CPU if GPU crashes or not available (handled by LightGBM usually if configured, but let's be explicit if needed)
# For now, I'll add 'device': 'gpu' because user has RTX 4070 and previous tasks used it. 
# BUT, LightGBM needs special compilation for GPU. If standard pip install lightgbm is used, it might fail with 'gpu'.
# To be safe and ensure "ca se lance", I will comment out 'device': 'gpu' appropriately or catch error?
# Actually, standard 'pip install lightgbm' on Windows often doesn't include GPU support out of the box unless installed with specific flags or from wheels.
# Using 'cpu' is safer for "make it run". User asked "fonctionne avec tout les depedances". 
# I will use 'cpu' to guarantee execution, or try 'gpu' and catch?
# No, let's stick to the code in the notebook exactly, which used default (CPU) implicitly.
# The notebook parameters didn't have 'device': 'gpu'.
# I'll stick to CPU to be safe, unless user asks for GPU.
del lgb_cfg['device'] 


def weighted_rmse_score(y_target, y_pred, w):
    y_target, y_pred, w = np.array(y_target), np.array(y_pred), np.array(w)
    denom = np.sum(w * (y_target ** 2))
    if denom <= 0:
        return 0.0
    numerator = np.sum(w * ((y_target - y_pred) ** 2))
    ratio = numerator / denom
    return float(np.sqrt(1.0 - np.clip(ratio, 0.0, 1.0)))

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

    fit_mask = tr_df.ts_index <= val_threshold
    val_mask = tr_df.ts_index > val_threshold

    X_fit = tr_df.loc[fit_mask, feature_cols]
    y_fit = tr_df.loc[fit_mask, 'y_target']
    w_fit = tr_df.loc[fit_mask, 'weight']

    X_hold = tr_df.loc[val_mask, feature_cols]
    y_hold = tr_df.loc[val_mask, 'y_target']
    w_hold = tr_df.loc[val_mask, 'weight']

    val_pred = np.zeros(len(y_hold))
    tst_pred = np.zeros(len(te_df))

    for seed in (42, 2024):
        mdl = lgb.LGBMRegressor(**lgb_cfg, random_state=seed)

        mdl.fit(
            X_fit,
            y_fit,
            sample_weight=w_fit,
            eval_set=[(X_hold, y_hold)],
            eval_sample_weight=[w_hold],
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )

        val_pred += mdl.predict(X_hold) / 2
        tst_pred += mdl.predict(te_df[feature_cols]) / 2

    cv_cache['y'].extend(y_hold.tolist())
    cv_cache['pred'].extend(val_pred.tolist())
    cv_cache['wt'].extend(w_hold.tolist())

    print(
        f"Horizon {hz} Score: "
        f"{weighted_rmse_score(y_hold, val_pred, w_hold):.5f}"
    )

    test_outputs.append(
        pd.DataFrame({'id': te_df['id'], 'prediction': tst_pred})
    )

    del tr_df, te_df
    gc.collect()

final_metric = weighted_rmse_score(cv_cache['y'], cv_cache['pred'], cv_cache['wt'])

print(
    f"\n{'='*40}\n"
    f"FINAL AGGREGATE SCORE: {final_metric:.6f}\n"
    f"{'='*40}"
)

pd.concat(test_outputs).to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
