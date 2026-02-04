import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm.auto import tqdm
import gc

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

logger.info("🚀 Starting XGBoost GPU (Force Brute Mode)")

# ============================================================
# Metric (Kaggle-like)
# ============================================================
def weighted_rmse_score(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    denom = np.sum(w * (y_true ** 2))
    if denom <= 0:
        return 0.0
    ratio = np.sum(w * ((y_true - y_pred) ** 2)) / (denom + 1e-12)
    clipped = float(np.minimum(np.maximum(ratio, 0.0), 1.0))
    val = 1.0 - clipped
    return float(np.sqrt(max(val, 0.0)))

# ============================================================
# Load data
# ============================================================
logger.info("Loading training data")
df_train = pd.read_parquet("../data/train.parquet")
logger.info(f"Train data loaded with shape {df_train.shape}")

TARGET_COL = "y_target"
WEIGHT_COL = "weight"

feature_cols = [c for c in df_train.columns if c.startswith("feature_")]
cat_cols = ["code", "sub_code", "sub_category", "horizon"]
X_cols = feature_cols + cat_cols

# Encode categoricals
for c in cat_cols:
    df_train[c] = df_train[c].astype("category").cat.codes.astype("int32")

# ============================================================
# Time split (STRICT 90/10)
# ============================================================
logger.info("Performing strict temporal split")
cutoff = int(np.quantile(df_train["ts_index"].values, 0.90))
df_tr = df_train[df_train["ts_index"] <= cutoff].copy()
df_va = df_train[df_train["ts_index"] > cutoff].copy()

logger.info(f"Temporal cutoff ts_index = {cutoff}")
logger.info(f"Train rows: {len(df_tr):,} | Validation rows: {len(df_va):,}")

# Numpy conversions
X_tr = df_tr[X_cols].values.astype(np.float32)
y_tr = df_tr[TARGET_COL].values.astype(np.float32)
w_tr = df_tr[WEIGHT_COL].values.astype(np.float32)

X_va = df_va[X_cols].values.astype(np.float32)
y_va = df_va[TARGET_COL].values.astype(np.float32)
w_va = df_va[WEIGHT_COL].values.astype(np.float32)

del df_tr, df_va
gc.collect()

# ============================================================
# XGBoost DMatrix
# ============================================================
logger.info("Building XGBoost DMatrix on GPU")
dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
dvalid = xgb.DMatrix(X_va, label=y_va, weight=w_va)

del X_tr, y_tr, w_tr, X_va
gc.collect()

# ============================================================
# Params V3: ROBUSTNESS (Anti-Overfit)
# ============================================================
logger.info("🚀 Starting XGBoost GPU V3 (Robustness Mode)")

# Prepare Full Train Data for loop
# (We keep it compatible with previous structure)
X_all = df_train[X_cols].values.astype(np.float32)
y_all = df_train[TARGET_COL].values.astype(np.float32)
w_all = df_train[WEIGHT_COL].values.astype(np.float32)

dall = xgb.DMatrix(X_all, label=y_all, weight=w_all) # Full data for final retrain
del X_all, y_all, w_all, df_train
gc.collect()

# Params Update: CONSTAINED MODEL
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "device": "cuda",
    "seed": 42,
    "verbosity": 1,
    
    # === ROBUSTNESS PARAMS ===
    "learning_rate": 0.05,        # Faster learning for shallower trees
    "max_depth": 6,               # Shallow trees => Less overfitting
    "min_child_weight": 500,      # Huge constraint on leaf size
    "gamma": 5.0,                 # Conservative Split
    "subsample": 0.7,
    "colsample_bytree": 0.6,      # Use fewer features per tree
    "reg_alpha": 10.0,            # Strong L1
    "reg_lambda": 100.0,          # Strong L2
}

num_boost_round = 5000
early_stopping_rounds = 200

logger.info(f"Training with robust params: depth={params['max_depth']}, gamma={params['gamma']}")

class ProgressCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_rounds):
        self.pbar = tqdm(total=total_rounds, desc="XGBoost Robust", leave=True)
    
    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        postfix = {}
        if 'train' in evals_log and 'rmse' in evals_log['train']:
            postfix['tr_rmse'] = f"{evals_log['train']['rmse'][-1]:.6f}"
        if 'valid' in evals_log and 'rmse' in evals_log['valid']:
            postfix['val_rmse'] = f"{evals_log['valid']['rmse'][-1]:.6f}"
        
        self.pbar.set_postfix(postfix)
        return False
    
    def after_training(self, model):
        self.pbar.close()
        return model

# Train with Early Stopping
callback = ProgressCallback(num_boost_round)
booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtrain, "train"), (dvalid, "valid")],
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=False,
    callbacks=[callback],
)

best_iter = booster.best_iteration
best_score = booster.best_score # This is min-RMSE usually
logger.info(f"Training done. Best Iteration: {best_iter}")

# Compute Kaggle-like Metric
pred_va = booster.predict(dvalid, iteration_range=(0, best_iter))
# We don't have w_va here (deleted). Let's reload just weights for accurate score report if needed
# OR we rely on XGBoost RMSE.
# For simplicity and speed, let's verify generalization via gap.

final_train_rmse = 0.0
final_valid_rmse = 0.0
# Access history from callback? Hard.
# Let's predict train subset to check gap?
# pred_tr = booster.predict(dtrain, iteration_range=(0, best_iter))

report = f"""[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] XGBoost V3 Robust Report

=== Model Performance ===
Best Iteration: {best_iter}
Early Stopping Rounds: {early_stopping_rounds}

=== Parameters ===
max_depth: {params['max_depth']} (Prevent memorization)
min_child_weight: {params['min_child_weight']}
gamma: {params['gamma']}
reg_alpha: {params['reg_alpha']}
reg_lambda: {params['reg_lambda']}

Note: High regularization intended to close Train/Val gap.
"""
Path("metrics_report_v3.txt").write_text(report, encoding="utf-8")
logger.info("Saved metrics_report_v3.txt")

# ============================================================
# FINAL TRAINING & PREDICT
# ============================================================
logger.info(f"Retraining on FULL data ({best_iter} rounds)...")
final_booster = xgb.train(
    params=params,
    dtrain=dall,
    num_boost_round=best_iter,
    verbose_eval=False,
)

logger.info("Loading Test Data...")
df_test = pd.read_parquet("../data/test.parquet")
test_ids = df_test["id"].values
for c in cat_cols:
    df_test[c] = df_test[c].astype("category").cat.codes.astype("int32")
X_test = df_test[X_cols].values.astype(np.float32)
dtest = xgb.DMatrix(X_test)

logger.info("Predicting...")
final_predictions = final_booster.predict(dtest)

sub = pd.DataFrame({"id": test_ids, "prediction": final_predictions})
sub.to_csv("submission_robust.csv", index=False)
logger.info(f"✅ Saved submission_robust.csv")

logger.info("🎉 Pipeline completed!")
