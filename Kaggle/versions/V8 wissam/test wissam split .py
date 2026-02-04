import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm.auto import tqdm

# ============================================================
# Logging configuration (PRO)
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

logger.info("Starting training pipeline")

# ============================================================
# Metric (Kaggle-like)
# ============================================================
def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))

def weighted_rmse_score(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """
    Kaggle-like score provided in your pipeline:
    score = sqrt( max( 1 - clip01( sum(w*(err^2))/sum(w*(y^2)) ), 0 ) )
    """
    denom = np.sum(w * (y_true ** 2))
    if denom <= 0:
        logger.warning("Denominator in metric is non-positive. Check target distribution.")
        return 0.0

    ratio = np.sum(w * ((y_true - y_pred) ** 2)) / (denom + 1e-12)
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return float(np.sqrt(max(val, 0.0)))

# ============================================================
# Reporting helpers (TXT export)
# ============================================================
def series_describe_str(arr: np.ndarray, name: str) -> str:
    """
    Produce a pandas-describe-like block with scientific notation, aligned,
    matching the format you showed.
    """
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
    extra_lines: list[str] | None = None,
) -> None:
    """
    Writes a metrics_report.txt containing:
    - timestamp
    - best_iteration
    - kaggle_like_score
    - weighted RMSE/MAE on validation
    - describe() stats for residuals, predictions, targets
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    residuals = y_true - y_pred

    w = weights.astype(float)
    wsum = float(np.sum(w)) + 1e-12
    wrmse = float(np.sqrt(np.sum(w * (residuals ** 2)) / wsum))
    wmae  = float(np.sum(w * np.abs(residuals)) / wsum)

    report: list[str] = []
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

# ============================================================
# Custom callback: tqdm progress bar + best score live
# ============================================================
def make_tqdm_callback(pbar, metric_name="rmse"):
    best = {"value": np.inf, "iter": 0}  # rmse: lower is better

    def _callback(env: lgb.callback.CallbackEnv):
        # env.evaluation_result_list: list of (data_name, eval_name, result, is_higher_better)
        if env.iteration == 0:
            pbar.reset(total=env.end_iteration)

        # update best from validation metric if available
        for data_name, eval_name, result, is_higher_better in env.evaluation_result_list:
            if data_name == "valid" and eval_name == metric_name:
                if (is_higher_better and result > best["value"]) or (not is_higher_better and result < best["value"]):
                    best["value"] = result
                    best["iter"] = env.iteration + 1  # iterations are 0-indexed internally

        # display current metrics + best
        postfix = {}
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            postfix[f"{data_name}_{eval_name}"] = f"{result:.6f}"

        postfix["best_iter"] = best["iter"]
        postfix[f"best_{metric_name}"] = f"{best['value']:.6f}"

        pbar.set_postfix(postfix)
        pbar.update(1)

    _callback.order = 10  # run after evaluation
    return _callback

# ============================================================
# Load data
# ============================================================
logger.info("Loading training data")
df_train = pd.read_parquet(r"C:\Users\Karim\Desktop\Kaggle\data\train.parquet")
logger.info(f"Train data loaded with shape {df_train.shape}")

TARGET_COL = "y_target"
WEIGHT_COL = "weight"

feature_cols = [c for c in df_train.columns if c.startswith("feature_")]
cat_cols = ["code", "sub_code", "sub_category", "horizon"]
X_cols = feature_cols + cat_cols

logger.info(f"Detected {len(feature_cols)} numerical features")
logger.info(f"Categorical features: {cat_cols}")

for c in cat_cols:
    df_train[c] = df_train[c].astype("category")

# ============================================================
# Time split (STRICT) — loop on multiple cutoffs
# ============================================================
split_quantiles = np.arange(0.88, 0.90, 0.001) # 10 values starting at 0.88 with step 0.001
ts = df_train["ts_index"].values

# Store metrics for global report
all_metrics_results = []

for q in split_quantiles:
    logger.info(f"\n==================== Split q={q:.2f} ====================")
    logger.info("Performing strict temporal split")
    cutoff = np.quantile(ts, q)  # (changed) no int-cast, more faithful split

    df_tr = df_train[df_train["ts_index"] <= cutoff].copy()
    df_va = df_train[df_train["ts_index"] > cutoff].copy()

    logger.info(f"Temporal cutoff ts_index = {cutoff}")
    logger.info(f"Train rows: {len(df_tr)} | Validation rows: {len(df_va)}")
    if df_va.empty:
        logger.error("Validation set is empty. Check ts_index distribution.")
        sys.exit(1)

    X_tr = df_tr[X_cols]
    y_tr = df_tr[TARGET_COL].values
    w_tr = df_tr[WEIGHT_COL].values

    X_va = df_va[X_cols]
    y_va = df_va[TARGET_COL].values
    w_va = df_va[WEIGHT_COL].values

    # ============================================================
    # LightGBM datasets (lgb.train API)
    # ============================================================
    logger.info("Building LightGBM datasets")
    dtrain = lgb.Dataset(
        X_tr, label=y_tr, weight=w_tr,
        categorical_feature=cat_cols, free_raw_data=False
    )
    dvalid = lgb.Dataset(
        X_va, label=y_va, weight=w_va,
        categorical_feature=cat_cols, free_raw_data=False
    )

    # ============================================================
    # Params
    # ============================================================
    params = dict(
    objective="regression",
    metric="rmse",
    boosting_type="gbdt",
    # device="gpu",           # GPU disabled: Incompatible with this dataset (Check failed: right_count > 0)
    
    learning_rate=0.01,
    num_leaves=1023,
    max_depth=-1,

    min_data_in_leaf=200,     # CPU default
    min_sum_hessian_in_leaf=1e-2,

    feature_fraction=0.7,
    bagging_fraction=0.7,
    bagging_freq=1,

    lambda_l1=0.1,
    lambda_l2=10.0,
    min_gain_to_split=0.0,

    max_bin=1024,             # CPU optimal

    extra_trees=True,         # CPU optimal
    path_smooth=2.0,

    num_threads=-1,
    seed=42,
    )

    num_boost_round = 5000
    early_stopping_rounds = 300
    
    # ============================================================
    # Train with LIVE progress
    # ============================================================
    logger.info("Starting training (live progress + best metric)")
    pbar = tqdm(total=num_boost_round, desc=f"LightGBM training q={q:.2f}", leave=True)

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        make_tqdm_callback(pbar, metric_name="rmse"),
    ]

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dvalid],
        valid_names=["valid"],
        num_boost_round=num_boost_round,
        callbacks=callbacks,
    )

    pbar.close()

    best_iter = booster.best_iteration
    logger.info(f"Training done. best_iteration = {best_iter}")

    # ============================================================
    # Validation (Kaggle-like)
    # ============================================================
    logger.info("Computing Kaggle-like validation score")
    pred_va = booster.predict(X_va, num_iteration=best_iter)
    score_va = weighted_rmse_score(y_va, pred_va, w_va)
    logger.info(f"[q={q:.2f}] Validation Kaggle-like score: {score_va:.6f}")

    # ============================================================
    # Export metrics report (.txt) — per split
    # ============================================================
    write_metrics_report_txt(
        f"metrics_report_q{int(q*100)}.txt",
        best_iter=best_iter,
        kaggle_like_score=score_va,
        y_true=y_va,
        y_pred=pred_va,
        weights=w_va,
        extra_lines=[
            f"Split quantile = {q}",
            f"Temporal cutoff ts_index = {cutoff}",
            f"Train rows = {len(df_tr)} | Val rows = {len(df_va)}",
            f"Num features = {len(feature_cols)} | Cat features = {len(cat_cols)}",
            "Report contains describe() for residuals / predictions / targets (validation).",
        ],
    )
    
    # Collect metrics for global report
    all_metrics_results.append({
        "quantile": q,
        "cutoff_ts": cutoff,
        "best_iter": best_iter,
        "kaggle_score": score_va,
        "train_rows": len(df_tr),
        "val_rows": len(df_va)
    })

# ============================================================
# Final Global Report
# ============================================================
logger.info("Generating global metrics report...")
global_report_path = "all_metrics_report.txt"
with open(global_report_path, "w", encoding="utf-8") as f:
    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📊 GLOBAL TRAINING REPORT\n")
    f.write("====================================================================\n")
    f.write(f"{'Quantile':<10} | {'Cutoff':<10} | {'Best Iter':<10} | {'Score (Kaggle)':<15} | {'Train Rows':<12} | {'Val Rows':<12}\n")
    f.write("-" * 85 + "\n")
    
    scores = []
    for res in all_metrics_results:
        scores.append(res["kaggle_score"])
        f.write(f"{res['quantile']:<10.3f} | {res['cutoff_ts']:<10.1f} | {res['best_iter']:<10} | {res['kaggle_score']:<15.6f} | {res['train_rows']:<12} | {res['val_rows']:<12}\n")
    
    f.write("-" * 85 + "\n")
    f.write(f"\nAverage Score: {np.mean(scores):.6f}\n")
    f.write(f"Best Score:    {np.min(scores):.6f} (at q={all_metrics_results[np.argmin(scores)]['quantile']:.3f})\n")
    f.write(f"Worst Score:   {np.max(scores):.6f}\n")

logger.info(f"Global metrics report saved to {global_report_path}")
