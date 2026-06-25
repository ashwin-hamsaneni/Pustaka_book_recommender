import pandas as pd
from pathlib import Path

# =====================================================
# PATHS
# =====================================================

RAW_DIR = Path("data/raw")
MERGED_DIR = Path("data/merged")

MERGED_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def standardize_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def standardize_ids(df):
    for col in df.columns:
        if col.endswith("_id"):
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
            )
    return df


def dataset_summary(df, name):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(df.head(3))


# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading datasets...")

users1 = pd.read_csv(RAW_DIR / "users.csv")
users2 = pd.read_csv(RAW_DIR / "users (1).csv")

books1 = pd.read_csv(RAW_DIR / "books.csv")
books2 = pd.read_csv(RAW_DIR / "books (1).csv")

transactions1 = pd.read_csv(RAW_DIR / "transactions.csv")
transactions2 = pd.read_csv(RAW_DIR / "transactions (1).csv")

# =====================================================
# STANDARDIZATION
# =====================================================

datasets = [
    users1, users2,
    books1, books2,
    transactions1, transactions2
]

for df in datasets:
    standardize_columns(df)
    standardize_ids(df)

# =====================================================
# SOURCE TRACKING
# =====================================================

users1["source"] = "dataset_1"
users2["source"] = "dataset_2"

books1["source"] = "dataset_1"
books2["source"] = "dataset_2"

transactions1["source"] = "dataset_1"
transactions2["source"] = "dataset_2"

# =====================================================
# PRE-MERGE SUMMARY
# =====================================================

dataset_summary(users1, "USERS DATASET 1")
dataset_summary(users2, "USERS DATASET 2")

dataset_summary(books1, "BOOKS DATASET 1")
dataset_summary(books2, "BOOKS DATASET 2")

dataset_summary(transactions1, "TRANSACTIONS DATASET 1")
dataset_summary(transactions2, "TRANSACTIONS DATASET 2")

# =====================================================
# MERGE DATASETS
# =====================================================

print("\nMerging datasets...")

users = pd.concat(
    [users1, users2],
    ignore_index=True
)

books = pd.concat(
    [books1, books2],
    ignore_index=True
)

transactions = pd.concat(
    [transactions1, transactions2],
    ignore_index=True
)

# =====================================================
# REMOVE DUPLICATES
# =====================================================

users_before = len(users)
books_before = len(books)
transactions_before = len(transactions)

users = users.drop_duplicates(
    subset=["user_id"]
)

books = books.drop_duplicates(
    subset=["book_id"]
)

transactions = transactions.drop_duplicates(
    subset=["transaction_id"]
)

users_removed = users_before - len(users)
books_removed = books_before - len(books)
transactions_removed = transactions_before - len(transactions)

# =====================================================
# COVERAGE CHECKS
# =====================================================

user_ids = set(users["user_id"])
book_ids = set(books["book_id"])

transaction_users = set(
    transactions["user_id"]
)

transaction_books = set(
    transactions["book_id"]
)

orphan_users = (
    transaction_users - user_ids
)

orphan_books = (
    transaction_books - book_ids
)

# =====================================================
# SAVE MERGED FILES
# =====================================================

users.to_csv(
    MERGED_DIR / "users_merged.csv",
    index=False
)

books.to_csv(
    MERGED_DIR / "books_merged.csv",
    index=False
)

transactions.to_csv(
    MERGED_DIR / "transactions_merged.csv",
    index=False
)

# =====================================================
# MERGE REPORT
# =====================================================

report = pd.DataFrame({

    "Metric": [

        "Users Before Merge",
        "Users After Merge",
        "Users Removed",

        "Books Before Merge",
        "Books After Merge",
        "Books Removed",

        "Transactions Before Merge",
        "Transactions After Merge",
        "Transactions Removed",

        "Unique Users",
        "Unique Books",

        "Orphan Users",
        "Orphan Books"

    ],

    "Value": [

        users_before,
        len(users),
        users_removed,

        books_before,
        len(books),
        books_removed,

        transactions_before,
        len(transactions),
        transactions_removed,

        users["user_id"].nunique(),
        books["book_id"].nunique(),

        len(orphan_users),
        len(orphan_books)

    ]
})

report.to_csv(
    MERGED_DIR / "merge_report.csv",
    index=False
)

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n")
print("=" * 60)
print("MERGE COMPLETE")
print("=" * 60)

print(f"\nUsers Shape        : {users.shape}")
print(f"Books Shape        : {books.shape}")
print(f"Transactions Shape : {transactions.shape}")

print("\nDuplicates Removed")
print("------------------")
print(f"Users        : {users_removed}")
print(f"Books        : {books_removed}")
print(f"Transactions : {transactions_removed}")

print("\nCoverage Checks")
print("------------------")
print(f"Orphan Users : {len(orphan_users)}")
print(f"Orphan Books : {len(orphan_books)}")

print("\nFiles Saved:")
print("data/merged/users_merged.csv")
print("data/merged/books_merged.csv")
print("data/merged/transactions_merged.csv")
print("data/merged/merge_report.csv")

print("\nPhase 1 Completed Successfully.")
