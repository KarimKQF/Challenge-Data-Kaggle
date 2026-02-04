import pandas as pd
import numpy as np

# Load both submissions
v9 = pd.read_csv(r'C:\Users\Karim\Desktop\PROJET LABS\Kaggle\V9\submission_v9_1_fixed.csv')
v7 = pd.read_csv(r'C:\Users\Karim\Desktop\PROJET LABS\Kaggle\V7 A submit\submission_v7_ultimate.csv')

print("="*60)
print("COMPARISON V9.1 vs V7")
print("="*60)

print("\n--- SHAPES ---")
print(f"V9.1: {v9.shape}")
print(f"V7:   {v7.shape}")

print("\n--- COLUMNS ---")
print(f"V9.1: {v9.columns.tolist()}")
print(f"V7:   {v7.columns.tolist()}")

print("\n--- V9.1 PREDICTION STATS ---")
print(f"Mean: {v9['prediction'].mean():.6f}")
print(f"Std:  {v9['prediction'].std():.6f}")
print(f"Min:  {v9['prediction'].min():.6f}")
print(f"Max:  {v9['prediction'].max():.6f}")
print(f"NaN:  {v9['prediction'].isna().sum()}")
print(f"Inf:  {np.isinf(v9['prediction']).sum()}")

print("\n--- V7 PREDICTION STATS ---")
print(f"Mean: {v7['prediction'].mean():.6f}")
print(f"Std:  {v7['prediction'].std():.6f}")
print(f"Min:  {v7['prediction'].min():.6f}")
print(f"Max:  {v7['prediction'].max():.6f}")
print(f"NaN:  {v7['prediction'].isna().sum()}")
print(f"Inf:  {np.isinf(v7['prediction']).sum()}")

print("\n--- ID COMPARISON ---")
v9_ids = set(v9['id'])
v7_ids = set(v7['id'])
print(f"V9 unique IDs: {len(v9_ids)}")
print(f"V7 unique IDs: {len(v7_ids)}")
print(f"Common IDs: {len(v9_ids & v7_ids)}")
print(f"V9 only: {len(v9_ids - v7_ids)}")
print(f"V7 only: {len(v7_ids - v9_ids)}")

print("\n--- SAMPLE IDs ---")
print("V9.1 first 5 IDs:", v9['id'].head(5).tolist())
print("V7 first 5 IDs:", v7['id'].head(5).tolist())

# Check if all predictions are zero or constant
print("\n--- ZERO/CONSTANT CHECK ---")
print(f"V9.1 all zeros: {(v9['prediction'] == 0).all()}")
print(f"V9.1 pct zeros: {(v9['prediction'] == 0).mean()*100:.2f}%")
print(f"V9.1 unique values: {v9['prediction'].nunique()}")

# Correlation check
merged = v9.merge(v7, on='id', suffixes=('_v9', '_v7'))
print("\n--- CORRELATION ---")
print(f"Correlation V9 vs V7: {merged['prediction_v9'].corr(merged['prediction_v7']):.4f}")
