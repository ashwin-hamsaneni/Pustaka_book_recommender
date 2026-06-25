import pandas as pd
import numpy as np
from pathlib import Path

# ======================================================
# PATHS
# ======================================================

MERGED_DIR = Path("data/merged")
CLEAN_DIR = Path("data/cleaned")

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# LOAD
# ======================================================

users = pd.read_csv(MERGED_DIR / "users_merged.csv")
books = pd.read_csv(MERGED_DIR / "books_merged.csv")
transactions = pd.read_csv(MERGED_DIR / "transactions_merged.csv")

print("\nStarting Phase 2 Cleaning")

# ======================================================
# CLEANING REPORT
# ======================================================

report = []

# ======================================================
# USERS
# ======================================================

users_before = len(users)

users["age"] = pd.to_numeric(
    users["age"],
    errors="coerce"
)

invalid_age_count = len(
    users[
        (users["age"] < 10) |
        (users["age"] > 90)
    ]
)

users.loc[
    (users["age"] < 10) |
    (users["age"] > 90),
    "age"
] = np.nan

missing_age_before = users["age"].isna().sum()

users["age"] = users["age"].fillna(
    users["age"].median()
)

for col in [
    "gender",
    "city",
    "state",
    "preferred_language",
    "subscription_type"
]:

    users[col] = (
        users[col]
        .astype(str)
        .str.strip()
        .str.title()
    )

    users[col] = users[col].replace(
        ["Nan", "None"],
        np.nan
    )

    users[col] = users[col].fillna(
        users[col].mode()[0]
    )

for col in [
    "signup_date",
    "subscription_start_date",
    "subscription_end_date"
]:

    users[col] = pd.to_datetime(
        users[col],
        errors="coerce"
    )

# ======================================================
# BOOKS
# ======================================================

books_before = len(books)

for col in [
    "genre",
    "language",
    "author",
    "format"
]:

    books[col] = (
        books[col]
        .astype(str)
        .str.strip()
        .str.title()
    )

    books[col] = books[col].replace(
        ["Nan", "None"],
        np.nan
    )

if "author" in books.columns:
    books["author"] = books["author"].fillna(
        "Unknown Author"
    )

numeric_cols = [
    "rating",
    "total_downloads",
    "page_count",
    "publication_year",
    "price"
]

for col in numeric_cols:

    books[col] = pd.to_numeric(
        books[col],
        errors="coerce"
    )

    books[col] = books[col].fillna(
        books[col].median()
    )

books.loc[
    books["publication_year"] < 1900,
    "publication_year"
] = np.nan

books.loc[
    books["publication_year"] > 2026,
    "publication_year"
] = np.nan

books["publication_year"] = (
    books["publication_year"]
    .fillna(
        books["publication_year"].median()
    )
)

# ======================================================
# TRANSACTIONS
# ======================================================

transactions_before = len(transactions)

transactions["transaction_date"] = pd.to_datetime(
    transactions["transaction_date"],
    errors="coerce"
)

future_transactions = len(
    transactions[
        transactions["transaction_date"]
        > pd.Timestamp.today()
    ]
)

transactions = transactions[
    transactions["transaction_date"]
    <= pd.Timestamp.today()
]

transactions["transaction_type"] = (
    transactions["transaction_type"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

transactions["transaction_type"] = (
    transactions["transaction_type"]
    .replace({
        "ebook_rental": "ebook_rental",
        "audiobook_rental": "audiobook_rental",
        "paperback_purchase": "paperback_purchase"
    })
)

# ======================================================
# REMOVE ORPHANS
# ======================================================

valid_users = set(users["user_id"])
valid_books = set(books["book_id"])

before_orphan = len(transactions)

transactions = transactions[
    transactions["user_id"].isin(valid_users)
]

transactions = transactions[
    transactions["book_id"].isin(valid_books)
]

orphans_removed = (
    before_orphan - len(transactions)
)

# ======================================================
# IQR OUTLIER TREATMENT
# ======================================================

q1 = users["age"].quantile(0.25)
q3 = users["age"].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

users["age"] = np.clip(
    users["age"],
    lower,
    upper
)

# ======================================================
# SAVE CLEAN DATA
# ======================================================

users.to_csv(
    CLEAN_DIR / "users_clean.csv",
    index=False
)

books.to_csv(
    CLEAN_DIR / "books_clean.csv",
    index=False
)

transactions.to_csv(
    CLEAN_DIR / "transactions_clean.csv",
    index=False
)

# ======================================================
# REPORT
# ======================================================

report.append([
    "Users",
    users_before,
    len(users)
])

report.append([
    "Books",
    books_before,
    len(books)
])

report.append([
    "Transactions",
    transactions_before,
    len(transactions)
])

cleaning_report = pd.DataFrame(
    report,
    columns=[
        "Dataset",
        "Rows Before",
        "Rows After"
    ]
)

cleaning_report.to_csv(
    CLEAN_DIR / "cleaning_report.csv",
    index=False
)

# ======================================================
# SUMMARY
# ======================================================

print("\nCleaning Complete")
print("=" * 50)

print(f"Invalid Ages Fixed      : {invalid_age_count}")
print(f"Missing Ages Filled     : {missing_age_before}")
print(f"Future Dates Removed    : {future_transactions}")
print(f"Orphan Records Removed  : {orphans_removed}")

print("\nFinal Shapes")
print(users.shape)
print(books.shape)
print(transactions.shape)

print("\nSaved To:")
print("data/cleaned/users_clean.csv")
print("data/cleaned/books_clean.csv")
print("data/cleaned/transactions_clean.csv")
print("data/cleaned/cleaning_report.csv")
