# check_pairs.py
import pandas as pd

match    = pd.read_csv("data/matchpairsDevTest.csv")
mismatch = pd.read_csv("data/mismatchpairsDevTest.csv")

print("=== matchpairsDevTest.csv ===")
print("Columns:", match.columns.tolist())
print(match.head(3))

print("\n=== mismatchpairsDevTest.csv ===")
print("Columns:", mismatch.columns.tolist())
print(mismatch.head(3))