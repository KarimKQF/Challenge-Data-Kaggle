import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
import warnings
import os

warnings.filterwarnings('ignore')

# Paths
TRAIN_PATH = r'c:\Users\Karim\Desktop\PROJET LABS\Kaggle\data\train.parquet'
TEST_PATH = r'c:\Users\Karim\Desktop\PROJET LABS\Kaggle\data\test.parquet'
VAL_THRESHOLD = 3500

def build_context_features(data, enc_stats=None):
    x = data.copy()
    group_cols = ['code', 'sub_code', 'sub_category', 'horizon']
    top_features = ['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'feature_s']

    # Sort to ensure groupby operations are consistent and faster
    x = x.sort_values(group_cols + ['ts_index'])

    # Encoded categorical signals
    if enc_stats is not None:
        for c in ['sub_category', 'sub_code']:
            x[c + '_enc'] = x[c].map(enc_stats[c]).fillna(enc_stats['global_mean']).astype(np.float32)

    # Interaction features
    x['d_al_am'] = (x['feature_al'] - x['feature_am']).astype(np.float32)
    x['r_al_am'] = (x['feature_al'] / (x['feature_am'] + 1e-7)).astype(np.float32)
    x['d_cg_by'] = (x['feature_cg'] - x['feature_by']).astype(np.float32)

    for col in top_features:
        if col not in x.columns:
            continue
        
        # Ensure base columns are float32 to save memory
        x[col] = x[col].astype(np.float32)

        for lag in [1, 3, 10]:
            x[f'{col}_lag{lag}'] = x.groupby(group_cols)[col].shift(lag).astype(np.float32)
   
        x[f'{col}_diff1'] = x.groupby(group_cols)[col].diff(1).astype(np.float32)
     
        for window in [5, 10]:
            # More memory efficient rolling on grouped data
            x[f'{col}_roll{window}'] = x.groupby(group_cols)[col].rolling(window, min_periods=1).mean().reset_index(level=[0,1,2,3], drop=True).astype(np.float32).values

            x[f'{col}_rollstd{window}'] = x.groupby(group_cols)[col].rolling(window, min_periods=1).std().reset_index(level=[0,1,2,3], drop=True).astype(np.float32).values

        x[f'{col}_ewm5'] = x.groupby(group_cols)[col].transform(
            lambda g: g.ewm(span=5, adjust=False).mean()
        ).astype(np.float32)

    # Temporal signal
    x['t_cycle'] = np.sin(2 * np.pi * x['ts_index'] / 100).astype(np.float32)

    return x

def weighted_rmse_score(y_target, y_pred, w):
    y_target, y_pred, w = np.array(y_target), np.array(y_pred), np.array(w)
    denom = np.sum(w * (y_target ** 2))
    if denom <= 0:
        return 0.0
    numerator = np.sum(w * ((y_target - y_pred) ** 2))
    ratio = numerator / denom
    return float(np.sqrt(1.0 - np.clip(ratio, 0.0, 1.0)))

def main():
    print("Loading data stats for encoding...")
    temp = pd.read_parquet(TRAIN_PATH, columns=['sub_category', 'sub_code', 'y_target', 'ts_index'])
    train_only = temp[temp.ts_index <= VAL_THRESHOLD]

    train_stats = {
        'sub_category': train_only.groupby('sub_category')['y_target'].mean().to_dict(),
        'sub_code': train_only.groupby('sub_code')['y_target'].mean().to_dict(),
        'global_mean': train_only['y_target'].mean()
    }
    del temp, train_only
    gc.collect()

    forecast_windows = [1, 3, 10, 25]
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
        'verbosity': -1
    }

    all_test_preds = []
    total_cv_y = []
    total_cv_pred = []
    total_cv_wt = []

    for hz in forecast_windows:
        print(f"\n>>> Training horizon = {hz}")
        
        tr_df = pd.read_parquet(TRAIN_PATH, filters=[('horizon', '==', hz)])
        tr_df = build_context_features(tr_df, train_stats)
        
        te_df = pd.read_parquet(TEST_PATH, filters=[('horizon', '==', hz)])
        te_df = build_context_features(te_df, train_stats)

        feature_cols = [c for c in tr_df.columns if c not in {
            'id', 'code', 'sub_code', 'sub_category', 'horizon', 'ts_index', 'weight', 'y_target'
        }]

        # Final cast to float32 to prevent upcasting during model fit
        tr_df[feature_cols] = tr_df[feature_cols].astype(np.float32)
        te_df[feature_cols] = te_df[feature_cols].astype(np.float32)

        fit_mask = tr_df.ts_index <= VAL_THRESHOLD
        val_mask = tr_df.ts_index > VAL_THRESHOLD

        X_fit = tr_df.loc[fit_mask, feature_cols]
        y_fit = tr_df.loc[fit_mask, 'y_target'].astype(np.float32)
        w_fit = tr_df.loc[fit_mask, 'weight'].astype(np.float32)
        X_val = tr_df.loc[val_mask, feature_cols]
        y_val = tr_df.loc[val_mask, 'y_target'].astype(np.float32)
        w_val = tr_df.loc[val_mask, 'weight'].astype(np.float32)

        hz_test_preds = np.zeros(len(te_df))
        hz_val_preds = np.zeros(len(y_val))

        for seed in [42, 2024]:
            print(f"  Seed {seed}...")
            lgb_cfg['seed'] = seed
            model = lgb.LGBMRegressor(**lgb_cfg)
            model.fit(
                X_fit, y_fit,
                sample_weight=w_fit,
                eval_set=[(X_val, y_val)],
                eval_sample_weight=[w_val],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(stopping_rounds=150), lgb.log_evaluation(period=500)]
            )
            hz_test_preds += model.predict(te_df[feature_cols]) / 2.0
            hz_val_preds += model.predict(X_val) / 2.0

        score = weighted_rmse_score(y_val, hz_val_preds, w_val)
        print(f"  Horizon {hz} Score: {score:.5f}")

        te_df['y_target'] = hz_test_preds
        all_test_preds.append(te_df[['id', 'y_target']])
        
        total_cv_y.extend(y_val.values)
        total_cv_pred.extend(hz_val_preds)
        total_cv_wt.extend(w_val.values)

        del tr_df, te_df, X_fit, y_fit, X_val, y_val, w_val
        gc.collect()

    final_score = weighted_rmse_score(total_cv_y, total_cv_pred, total_cv_wt)
    print(f"\nFinal Weighted CV Score: {final_score:.5f}")

    submission = pd.concat(all_test_preds).sort_values('id')
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission saved to submission.csv")

if __name__ == "__main__":
    main()
