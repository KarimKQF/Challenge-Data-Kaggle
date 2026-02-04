"""
================================================================================
KAGGLE SOLUTION: LightGBM OPTIMIZED (Based on User's Training Template)
================================================================================
This script combines:
- The compliance & feature engineering from solution_compliant_gpu.py
- The LightGBM training approach with live tqdm progress bar
- Strict temporal validation for robust early stopping
- Final model retrained on ALL data for maximum score


================================================================================
"""

import gc
import os
import sys
import time
import warnings
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

# Suppress Warnings
warnings.filterwarnings('ignore')

# ============================================================
# Logging configuration
# ============================================================
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    # Use absolute paths to the data
    KAGGLE_DIR = r"c:\Users\Karim\Desktop\Kaggle"
    TRAIN_PATH = os.path.join(KAGGLE_DIR, 'ts-forecasting', 'data', 'train.parquet')
    TEST_PATH = os.path.join(KAGGLE_DIR, 'ts-forecasting', 'data', 'test.parquet')
    # Output to Kaggle directory
    OUTPUT_DIR = KAGGLE_DIR
    SUB_PATH = os.path.join(OUTPUT_DIR, 'submission.csv')
    METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics_report.txt')
    
    # Feature Engineering (EXTENDED)
    LAGS = [1, 2, 3, 5, 7, 10, 14, 21]
    ROLLING_WINDOWS = [3, 5, 10, 20, 50]
    
    # Temporal Split (strict)
    VAL_QUANTILE = 0.90
    
    # Seed Averaging for stability
    SEEDS = [42, 123, 456]
    
    # LightGBM Params (ULTIMATE - RTX 4070 GPU)
    LGB_PARAMS = dict(
        objective="regression",
        learning_rate=0.02,           # Lower LR = more precision
        num_leaves=127,               # Deeper trees
        max_depth=12,                 # Control depth
        min_data_in_leaf=100,         # Less constraint
        feature_fraction=0.7,         # More diversity
        bagging_fraction=0.85,        # Slightly more data
        bagging_freq=1,
        lambda_l1=0.1,                # L1 reg
        lambda_l2=1.5,                # Stronger L2
        min_gain_to_split=0.01,       # Prevent overfitting
        force_col_wise=True,
        num_threads=-1,
        verbose=-1,
        # === GPU ACCELERATION ===
        device="gpu",                 # Use GPU (OpenCL)
        gpu_platform_id=0,            # First GPU platform
        gpu_device_id=0,              # RTX 4070
        gpu_use_dp=True,              # Double precision for accuracy
        max_bin=255,                  # Optimal for GPU
    )
    
    NUM_BOOST_ROUND = 15000           # More rounds with low LR
    EARLY_STOPPING_ROUNDS = 300       # Patient early stopping

CFG = Config()

# ==============================================================================
# METRICS
# ==============================================================================

def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))

def weighted_rmse_score(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """Kaggle-like competition score."""
    denom = np.sum(w * (y_true ** 2))
    if denom <= 0:
        logger.warning("Denominator in metric is non-positive.")
        return 0.0
    ratio = np.sum(w * ((y_true - y_pred) ** 2)) / (denom + 1e-12)
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return float(np.sqrt(max(val, 0.0)))

# ==============================================================================
# REPORTING
# ==============================================================================

def series_describe_str(arr: np.ndarray, name: str) -> str:
    s = pd.Series(arr, name=name)
    desc = s.describe()
    lines = [f"📊 {name} Statistics:"]
    for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        v = float(desc.loc[k])
        lines.append(f"{k:<7} {v: .6e}")
    return "\n".join(lines)

def write_metrics_report_txt(
    filepath: str,
    *,
    best_iter: int,
    kaggle_like_score: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    extra_lines: list = None,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    residuals = y_true - y_pred

    w = weights.astype(float)
    wsum = float(np.sum(w)) + 1e-12
    wrmse = float(np.sqrt(np.sum(w * (residuals ** 2)) / wsum))
    wmae = float(np.sum(w * np.abs(residuals)) / wsum)

    report = []
    report.append(f"[{ts}] ✅ Training report")
    report.append("")
    report.append("=== Key metrics ===")
    report.append(f"best_iteration: {best_iter}")
    report.append(f"kaggle_like_score: {kaggle_like_score:.6f}")
    report.append(f"weighted_RMSE: {wrmse:.6f}")
    report.append(f"weighted_MAE: {wmae:.6f}")

    report.append("")
    report.append(series_describe_str(residuals, "Residuals (y_true - y_pred)"))
    report.append("")
    report.append(series_describe_str(y_pred, "Predictions (val)"))
    report.append("")
    report.append(series_describe_str(y_true, "Targets (val)"))

    if extra_lines:
        report.append("")
        report.append("=== Notes ===")
        report.extend(extra_lines)

    Path(filepath).write_text("\n".join(report), encoding="utf-8")
    logger.info(f"Saved metrics report to {filepath}")

# ==============================================================================
# TQDM CALLBACK
# ==============================================================================

def make_tqdm_callback(pbar, metric_name="rmse"):
    best = {"value": np.inf, "iter": 0}

    def _callback(env: lgb.callback.CallbackEnv):
        if env.iteration == 0:
            pbar.reset(total=env.end_iteration)

        for data_name, eval_name, result, is_higher_better in env.evaluation_result_list:
            if data_name == "valid" and eval_name == metric_name:
                if (is_higher_better and result > best["value"]) or (not is_higher_better and result < best["value"]):
                    best["value"] = result
                    best["iter"] = env.iteration + 1

        postfix = {}
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            postfix[f"{data_name}_{eval_name}"] = f"{result:.6f}"

        postfix["best_iter"] = best["iter"]
        postfix[f"best_{metric_name}"] = f"{best['value']:.6f}"

        pbar.set_postfix(postfix)
        pbar.update(1)

    _callback.order = 10
    return _callback

# ==============================================================================
# FEATURE ENGINE
# ==============================================================================

class FeatureEngine:
    def __init__(self):
        self.cat_cols = ['code', 'sub_code', 'sub_category']
        self.encoders = {}
        self.ts_min = 0.0
        self.ts_max = 1.0
        self.global_mean = 0.0
        self.last_y = None
        self.last_rolls = {}
        self.means_L2 = None
        self.means_L4 = None

    def fit(self, train_df):
        logger.info("Fitting Feature Engine...")
        
        for col in self.cat_cols:
            le = LabelEncoder()
            unique_vals = train_df[col].astype(str).unique().tolist()
            le.fit(unique_vals + ['__UNK__'])
            self.encoders[col] = le
            
        self.ts_min = float(train_df['ts_index'].min())
        self.ts_max = float(train_df['ts_index'].max())
        self.global_mean = float(train_df['y_target'].mean())
        
        # Hierarchical Means
        mean_l2 = train_df.groupby(['code', 'sub_code', 'sub_category'], observed=True)['y_target'].mean()
        idx_l2 = [mean_l2.index.get_level_values(i).astype(str) for i in range(3)]
        self.means_L2 = pd.Series(mean_l2.values, index=pd.MultiIndex.from_arrays(idx_l2, names=mean_l2.index.names))
        
        mean_l4 = train_df.groupby(['code'], observed=True)['y_target'].mean()
        self.means_L4 = pd.Series(mean_l4.values, index=mean_l4.index.astype(str))
        
        # Last Y per group
        grp_cols = ['code', 'sub_code', 'sub_category', 'horizon']
        last_y = train_df.groupby(grp_cols, observed=True)['y_target'].last()
        
        idx_arrays = [last_y.index.get_level_values(i).astype(str) if n in self.cat_cols else last_y.index.get_level_values(i).astype(np.int32) 
                      for i, n in enumerate(last_y.index.names)]
        self.last_y = pd.Series(last_y.values, index=pd.MultiIndex.from_arrays(idx_arrays, names=last_y.index.names))
        
        # Rolling means at end of train
        max_w = max(CFG.ROLLING_WINDOWS)
        tail_rows = train_df.groupby(grp_cols, observed=True).tail(max_w)[grp_cols + ['y_target']]
        
        for w in CFG.ROLLING_WINDOWS:
            val = tail_rows.groupby(grp_cols, observed=True)['y_target'].apply(lambda x: x.tail(w).mean())
            idx_av = [val.index.get_level_values(i).astype(str) if n in self.cat_cols else val.index.get_level_values(i).astype(np.int32) 
                      for i, n in enumerate(val.index.names)]
            val_typed = pd.Series(val.values, index=pd.MultiIndex.from_arrays(idx_av, names=val.index.names))
            self.last_rolls[w] = val_typed.reindex(self.last_y.index).astype(np.float32)
            
        del tail_rows, mean_l2, mean_l4, last_y
        gc.collect()

    def transform_train_inplace(self, df):
        logger.info("Transforming Train...")
        grp_cols = ['code', 'sub_code', 'sub_category', 'horizon']
        
        for col in self.cat_cols:
            df[f'{col}_enc'] = self.encoders[col].transform(df[col].astype(str)).astype(np.int32)
            
        df['ts_norm'] = ((df['ts_index'] - self.ts_min) / (self.ts_max - self.ts_min + 1e-5)).astype(np.float32)
        
        g = df.groupby(grp_cols, observed=True)['y_target']
        for lag in CFG.LAGS:
            df[f'y_lag{lag}'] = g.shift(lag).astype(np.float32)
            
        df['y_shifted_1'] = g.shift(1).astype(np.float32)
        g_s1 = df.groupby(grp_cols, observed=True)['y_shifted_1']
        for w in CFG.ROLLING_WINDOWS:
            df[f'y_roll{w}_mean'] = g_s1.transform(lambda x: x.rolling(w, min_periods=1).mean()).astype(np.float32)
        df.drop(columns=['y_shifted_1'], inplace=True)
        gc.collect()

    def transform_test_inplace(self, df):
        logger.info("Transforming Test...")
        grp_cols = ['code', 'sub_code', 'sub_category', 'horizon']
        
        for col in self.cat_cols:
            col_str = df[col].astype(str)
            unknowns = ~col_str.isin(self.encoders[col].classes_)
            if unknowns.any():
                col_str = col_str.copy()
                col_str[unknowns] = '__UNK__'
            df[f'{col}_enc'] = self.encoders[col].transform(col_str).astype(np.int32)
            
        df['ts_norm'] = ((df['ts_index'] - self.ts_min) / (self.ts_max - self.ts_min + 1e-5)).astype(np.float32)
        
        # Keys for mapping
        idx_arrays = [df[c].astype(str) for c in self.cat_cols] + [df['horizon'].astype(np.int32)]
        keys_primary = pd.MultiIndex.from_arrays(idx_arrays, names=grp_cols)
        
        l2_cols = ['code', 'sub_code', 'sub_category']
        idx_l2 = [df[c].astype(str) for c in l2_cols]
        keys_l2 = pd.MultiIndex.from_arrays(idx_l2, names=l2_cols)
        keys_l3 = df['code'].astype(str)

        def apply_hierarchical(target_col, source_primary, source_l2, source_l3, default_val):
            vals = source_primary.reindex(keys_primary).to_numpy(dtype=np.float32)
            mask = np.isnan(vals)
            if mask.sum() > 0:
                fill_l2 = source_l2.reindex(keys_l2).to_numpy(dtype=np.float32)
                vals[mask] = fill_l2[mask]
                mask = np.isnan(vals)
                if mask.sum() > 0:
                    fill_l3 = keys_l3.map(source_l3).to_numpy(dtype=np.float32)
                    vals[mask] = fill_l3[mask]
                    mask = np.isnan(vals)
                    if mask.any():
                        vals[mask] = default_val
            df[target_col] = vals
            
        for lag in CFG.LAGS:
            apply_hierarchical(f'y_lag{lag}', self.last_y, self.means_L2, self.means_L4, self.global_mean)
            
        for w in CFG.ROLLING_WINDOWS:
            apply_hierarchical(f'y_roll{w}_mean', self.last_rolls[w], self.means_L2, self.means_L4, self.global_mean)
            
        gc.collect()

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    logger.info("=" * 60)
    logger.info("STARTING LightGBM OPTIMIZED PIPELINE")
    logger.info("=" * 60)
    
    # 1. Load Train
    logger.info("Loading training data...")
    train = pd.read_parquet(CFG.TRAIN_PATH)
    logger.info(f"Train data loaded: {train.shape}")
    
    # Enforce dtypes
    for c in ['code', 'sub_code', 'sub_category']:
        if train[c].dtype.name != 'category':
            train[c] = train[c].astype(str).astype('category')
    if 'horizon' in train.columns:
        train['horizon'] = train['horizon'].astype(np.int32)
    if 'ts_index' in train.columns:
        train['ts_index'] = train['ts_index'].astype(np.int32)
    for c in train.select_dtypes(include=['float64']).columns:
        train[c] = train[c].astype(np.float32)
    
    # Sort
    logger.info("Sorting training data...")
    train.sort_values(['code', 'sub_code', 'sub_category', 'horizon', 'ts_index'], inplace=True)
    train.reset_index(drop=True, inplace=True)
    
    train_min_target = train['y_target'].min()
    
    # Feature Engineering
    fe = FeatureEngine()
    fe.fit(train)
    fe.transform_train_inplace(train)
    
    # Define columns
    feature_cols = [c for c in train.columns if c.startswith('feature_')]
    fe_cols = [c for c in train.columns if c.endswith('_enc') or c.startswith('y_lag') or c.startswith('y_roll') or c == 'ts_norm']
    X_cols = feature_cols + fe_cols
    cat_cols = ['code', 'sub_code', 'sub_category', 'horizon']
    
    logger.info(f"Numerical features: {len(feature_cols)} | Engineered features: {len(fe_cols)}")
    
    # Make categoricals for LightGBM (encoded versions)
    lgb_cat_cols = [f'{c}_enc' for c in ['code', 'sub_code', 'sub_category']]
    for c in lgb_cat_cols:
        train[c] = train[c].astype('category')
    
    # 2. Temporal Split
    logger.info("Performing strict temporal split...")
    cutoff = int(np.quantile(train["ts_index"].values, CFG.VAL_QUANTILE))
    df_tr = train[train["ts_index"] <= cutoff].copy()
    df_va = train[train["ts_index"] > cutoff].copy()
    
    logger.info(f"Temporal cutoff ts_index = {cutoff}")
    logger.info(f"Train rows: {len(df_tr)} | Validation rows: {len(df_va)}")
    
    if df_va.empty:
        logger.error("Validation set is empty!")
        sys.exit(1)
    
    X_tr = df_tr[X_cols]
    y_tr = df_tr['y_target'].values
    w_tr = df_tr['weight'].values
    
    X_va = df_va[X_cols]
    y_va = df_va['y_target'].values
    w_va = df_va['weight'].values
    
    # 3. LightGBM Datasets
    logger.info("Building LightGBM datasets...")
    dtrain = lgb.Dataset(
        X_tr, label=y_tr, weight=w_tr,
        categorical_feature=lgb_cat_cols, free_raw_data=False
    )
    dvalid = lgb.Dataset(
        X_va, label=y_va, weight=w_va,
        categorical_feature=lgb_cat_cols, free_raw_data=False
    )
    
    # 4. Train with LIVE progress
    logger.info("Starting training (live progress + best metric)...")
    pbar = tqdm(total=CFG.NUM_BOOST_ROUND, desc="LightGBM training", leave=True)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=CFG.EARLY_STOPPING_ROUNDS, verbose=False),
        make_tqdm_callback(pbar, metric_name="rmse"),
    ]
    
    booster = lgb.train(
        params=CFG.LGB_PARAMS,
        train_set=dtrain,
        valid_sets=[dvalid],
        valid_names=["valid"],
        num_boost_round=CFG.NUM_BOOST_ROUND,
        callbacks=callbacks,
    )
    
    pbar.close()
    
    best_iter = booster.best_iteration
    logger.info(f"Training done. best_iteration = {best_iter}")
    
    # 5. Validation Score
    logger.info("Computing Kaggle-like validation score...")
    pred_va = booster.predict(X_va, num_iteration=best_iter)
    score_va = weighted_rmse_score(y_va, pred_va, w_va)
    logger.info(f"🎯 Validation Kaggle-like score: {score_va:.6f}")
    
    # 6. Export metrics report
    write_metrics_report_txt(
        CFG.METRICS_PATH,
        best_iter=best_iter,
        kaggle_like_score=score_va,
        y_true=y_va,
        y_pred=pred_va,
        weights=w_va,
        extra_lines=[
            f"Temporal cutoff ts_index = {cutoff}",
            f"Train rows = {len(df_tr)} | Val rows = {len(df_va)}",
            f"Num features = {len(X_cols)}",
        ],
    )
    
    # 7. Free memory before final training
    del df_tr, df_va, dtrain, dvalid, booster, pred_va
    gc.collect()
    
    # 8. Train FINAL model on ALL data (single seed, memory-efficient)
    logger.info("=" * 50)
    logger.info("� Training FINAL model on ALL data...")
    logger.info("=" * 50)
    
    X_all = train[X_cols].values.astype(np.float32)  # Convert to numpy for memory efficiency
    y_all = train['y_target'].values.astype(np.float32)
    w_all = train['weight'].values.astype(np.float32)
    
    # Free train DataFrame
    test_ids = None  # Will load later
    del train
    gc.collect()
    
    dall = lgb.Dataset(
        X_all, label=y_all, weight=w_all,
        free_raw_data=True  # No categorical_feature needed - data is numpy array with encoded integers
    )
    
    params_final = CFG.LGB_PARAMS.copy()
    params_final['seed'] = 42
    
    pbar_final = tqdm(total=best_iter, desc="Final Model", leave=True)
    
    def simple_tqdm_callback(env):
        pbar_final.update(1)
    simple_tqdm_callback.order = 10
    
    final_booster = lgb.train(
        params=params_final,
        train_set=dall,
        num_boost_round=best_iter,
        callbacks=[simple_tqdm_callback],
    )
    
    pbar_final.close()
    
    del dall, X_all, y_all, w_all
    gc.collect()
    
    # 9. Load Test & Transform
    logger.info("Loading test data...")
    test = pd.read_parquet(CFG.TEST_PATH)
    logger.info(f"Test data loaded: {test.shape}")
    
    for c in ['code', 'sub_code', 'sub_category']:
        if test[c].dtype.name != 'category':
            test[c] = test[c].astype(str).astype('category')
    if 'horizon' in test.columns:
        test['horizon'] = test['horizon'].astype(np.int32)
    if 'ts_index' in test.columns:
        test['ts_index'] = test['ts_index'].astype(np.int32)
    for c in test.select_dtypes(include=['float64']).columns:
        test[c] = test[c].astype(np.float32)
    
    fe.transform_test_inplace(test)
    
    for c in lgb_cat_cols:
        test[c] = test[c].astype('category')
    
    # 10. Predict
    logger.info("Generating test predictions...")
    X_test = test[X_cols].values.astype(np.float32)
    test_pred = final_booster.predict(X_test)
    
    logger.info(f"   Pred range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    
    # Handle NaNs
    if np.isnan(test_pred).any():
        logger.warning(f"Found {np.isnan(test_pred).sum()} NaNs in predictions. Filling with 0.")
        test_pred = np.nan_to_num(test_pred, nan=0.0)
    
    # Conditional clipping
    if train_min_target >= 0:
        logger.info("Clipping predictions to 0 (target >= 0)")
        test_pred = test_pred.clip(0, None)
    else:
        logger.info("No clipping applied (target has negatives)")
    
    # 11. Save Submission
    sub = pd.DataFrame({'id': test['id'], 'prediction': test_pred})
    sub.to_csv(CFG.SUB_PATH, index=False)
    logger.info(f"✅ Saved submission.csv to {CFG.SUB_PATH} with shape {sub.shape}")
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
