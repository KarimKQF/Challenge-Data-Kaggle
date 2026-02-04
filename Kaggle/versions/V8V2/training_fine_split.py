"""
🚀 FINE-TUNED SPLIT ANALYSIS (5 Splits)
Based on V8 logic.
Range: [0.896, 0.898] with 5 splits.

Output:
- metrics_fine_split.txt
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm.auto import tqdm

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

logger.info("🚀 Starting FINE-TUNED split analysis pipeline")

# ============================================================
# Paths
# ============================================================
DATA_DIR = Path(r"C:\Users\Karim\Desktop\Kaggle\data")
OUTPUT_DIR = Path(r"C:\Users\Karim\Desktop\Kaggle")
METRICS_FILE = OUTPUT_DIR / "metrics_fine_split.txt"

# ============================================================
# Metric
# ============================================================
def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))

def weighted_rmse_score(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    denom = np.sum(w * (y_true ** 2))
    if denom <= 0: return 0.0
    ratio = np.sum(w * ((y_true - y_pred) ** 2)) / (denom + 1e-12)
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return float(np.sqrt(max(val, 0.0)))

def series_describe_str(arr: np.ndarray, name: str) -> str:
    s = pd.Series(arr, name=name)
    desc = s.describe()
    lines = [f"📊 {name} Statistics:"]
    for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        v = float(desc.loc[k])
        lines.append(f"{k:<7} {v: .6e}")
    return "\n".join(lines)

# ============================================================
# Load Data
# ============================================================
logger.info("Loading training data...")
df_train = pd.read_parquet(DATA_DIR / "train.parquet")
logger.info(f"Train data loaded: {df_train.shape}")

TARGET_COL = "y_target"
WEIGHT_COL = "weight"

feature_cols = [c for c in df_train.columns if c.startswith("feature_")]
cat_cols = ["code", "sub_code", "sub_category", "horizon"]
X_cols = feature_cols + cat_cols

for c in cat_cols:
    df_train[c] = df_train[c].astype("category")

# ============================================================
# SPLIT CONFIGURATION (5 Splits between 0.896 and 0.898)
# ============================================================
split_quantiles = np.linspace(0.896, 0.898, 5)
ts = df_train["ts_index"].values

logger.info(f"Testing {len(split_quantiles)} splits: {split_quantiles}")

# Initialize report
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    f.write(f"[{datetime.now()}] 🚀 FINE-TUNED SPLIT REPORT (q=0.896 -> 0.898)\n")
    f.write("====================================================================\n")
    f.write(f"{'Quantile':<10} | {'Cutoff':<10} | {'Best Iter':<10} | {'Score':<15} | {'Train Rows':<12} | {'Val Rows':<12}\n")
    f.write("--------------------------------------------------------------------\n")

# ============================================================
# LOOP OVER SPLITS
# ============================================================
for q in split_quantiles:
    logger.info(f"\n==================== Split q={q:.4f} ====================")
    cutoff = np.quantile(ts, q)
    
    df_tr = df_train[df_train["ts_index"] <= cutoff].copy()
    df_va = df_train[df_train["ts_index"] > cutoff].copy()
    
    logger.info(f"Cutoff: {cutoff} | Train: {len(df_tr)} | Val: {len(df_va)}")
    
    if df_va.empty:
        logger.error("Validation set empty! Skipping.")
        continue

    X_tr = df_tr[X_cols]
    y_tr = df_tr[TARGET_COL].values
    w_tr = df_tr[WEIGHT_COL].values

    X_va = df_va[X_cols]
    y_va = df_va[TARGET_COL].values
    w_va = df_va[WEIGHT_COL].values

    # LightGBM Datasets
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, categorical_feature=cat_cols, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va, weight=w_va, categorical_feature=cat_cols, free_raw_data=False)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 1023,
        "max_depth": -1,
        "min_data_in_leaf": 200,
        "min_sum_hessian_in_leaf": 1e-2,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 10.0,
        "min_gain_to_split": 0.0,
        "max_bin": 1024,
        "extra_trees": True,
        "path_smooth": 2.0,
        "num_threads": -1,
        "seed": 42,
        "verbosity": -1,
    }

    num_boost_round = 5000
    early_stopping_rounds = 300
    
    # Training
    pbar = tqdm(total=num_boost_round, desc=f"LGBM q={q:.4f}", leave=True)
    best_score_holder = [float("inf")]
    
    def callback_progress(env):
        pbar.update(1)
        if env.evaluation_result_list:
            score = env.evaluation_result_list[0][2]
            if score < best_score_holder[0]:
                best_score_holder[0] = score
            pbar.set_postfix({"rmse": f"{score:.6f}", "bst": f"{best_score_holder[0]:.6f}"})

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            callback_progress,
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        ],
    )
    pbar.close()
    
    best_iter = model.best_iteration
    
    # Compute Kaggle Score
    pred_val = model.predict(X_va[X_cols], num_iteration=best_iter)
    kaggle_score = weighted_rmse_score(y_va, pred_val, w_va)
    
    logger.info(f"✅ Finished q={q:.4f} | Kaggle Score: {kaggle_score:.6f}")
    
    # Append to report
    with open(METRICS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{q:<10.4f} | {cutoff:<10} | {best_iter:<10} | {kaggle_score:<15.6f} | {len(df_tr):<12} | {len(df_va):<12}\n")

logger.info(f"🎉 Analysis complete. Results in {METRICS_FILE}")
