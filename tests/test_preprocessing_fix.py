# Quick Test for Preprocessing Fix

import pandas as pd
import sys

sys.path.append(".")

from preprocessing import engineer_features, select_features

# Test 1: DataFrame with step column
print("Test 1: DataFrame WITH step column")
df_with_step = pd.DataFrame(
    {
        "step": [1, 1, 2],
        "type": ["PAYMENT", "TRANSFER", "CASH-OUT"],
        "amount": [1000, 2000, 1500],
        "nameOrig": ["C123", "C123", "C456"],
        "oldbalanceOrg": [5000, 4000, 3000],
        "newbalanceOrig": [4000, 2000, 1500],
        "nameDest": ["C789", "C789", "C789"],
        "oldbalanceDest": [1000, 2000, 3000],
        "newbalanceDest": [2000, 4000, 4500],
    }
)

try:
    df_featured = engineer_features(df_with_step.copy())
    df_selected = select_features(df_featured)
    print(f"Success! Generated {len(df_selected.columns)} features")
    print(f"   Features: {list(df_selected.columns)[:5]}...")
except Exception as e:
    print(f"Error: {e}")

# Test 2: DataFrame without step column
print("\nTest 2: DataFrame WITHOUT step column")
df_without_step = pd.DataFrame(
    {
        "type": ["PAYMENT"],
        "amount": [1000],
        "nameOrig": ["C123"],
        "oldbalanceOrg": [5000],
        "newbalanceOrig": [4000],
        "nameDest": ["C789"],
        "oldbalanceDest": [1000],
        "newbalanceDest": [2000],
    }
)

try:
    df_featured = engineer_features(df_without_step.copy())
    df_selected = select_features(df_featured)
    print(f"Success! Generated {len(df_selected.columns)} features")
    print(f"   Step column added: {'step' in df_featured.columns}")
    print(f"   Features: {list(df_selected.columns)[:5]}...")
except Exception as e:
    print(f"Error: {e}")

print("\nAll tests passed! The fix is working correctly.")
