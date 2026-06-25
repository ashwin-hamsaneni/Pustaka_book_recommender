import pandas as pd
import numpy as np

from pathlib import Path

# =====================================================
# PATHS
# =====================================================

FEATURE_DIR = Path("data/features")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# LOAD
# =====================================================

interactions = pd.read_csv(
    FEATURE_DIR /
    "interaction_features.csv"
)

interactions["transaction_date"] = pd.to_datetime(
    interactions["transaction_date"]
)

print("Interactions Loaded")
print(interactions.shape)

# =====================================================
# SORT BY TIME
# =====================================================

interactions = interactions.sort_values(
    by="transaction_date"
)

# =====================================================
# TIME SPLIT
# =====================================================

split_index = int(
    len(interactions) * 0.8
)

cutoff_date = interactions.iloc[split_index]["transaction_date"]

train = interactions[
    interactions["transaction_date"] < cutoff_date
]

test = interactions[
    interactions["transaction_date"] >= cutoff_date
]

print("Cutoff Date:", cutoff_date)
# =====================================================
# INTERACTION MATRIX
# =====================================================

interaction_matrix = (
    train.pivot_table(
        index="user_id",
        columns="book_id",
        values="interaction_weight",
        aggfunc="sum",
        fill_value=0
    )
)

# =====================================================
# SPARSITY
# =====================================================

non_zero = np.count_nonzero(
    interaction_matrix.values
)

total = (
    interaction_matrix.shape[0]
    *
    interaction_matrix.shape[1]
)

density = non_zero / total
sparsity = 1 - density

# =====================================================
# SAVE
# =====================================================

train.to_csv(
    PROCESSED_DIR /
    "train_interactions.csv",
    index=False
)

test.to_csv(
    PROCESSED_DIR /
    "test_interactions.csv",
    index=False
)

interaction_matrix.to_csv(
    PROCESSED_DIR /
    "interaction_matrix.csv"
)

summary = pd.DataFrame({

    "Metric": [

        "Train Rows",
        "Test Rows",

        "Users",
        "Books",

        "Density",
        "Sparsity"

    ],

    "Value": [

        len(train),
        len(test),

        interaction_matrix.shape[0],
        interaction_matrix.shape[1],

        density,
        sparsity

    ]
})

summary.to_csv(
    PROCESSED_DIR /
    "split_summary.csv",
    index=False
)

# =====================================================
# OUTPUT
# =====================================================

print("\nSplit Complete")

print(
    "Train:",
    train.shape
)

print(
    "Test:",
    test.shape
)

print(
    "Interaction Matrix:",
    interaction_matrix.shape
)

print(
    f"Density: {density:.6f}"
)

print(
    f"Sparsity: {sparsity:.6f}"
)
