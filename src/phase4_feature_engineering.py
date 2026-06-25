import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# PATHS
# =====================================================

CLEAN_DIR = Path("data/cleaned")
FEATURE_DIR = Path("data/features")

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# LOAD
# =====================================================

users = pd.read_csv(
    CLEAN_DIR / "users_clean.csv"
)

books = pd.read_csv(
    CLEAN_DIR / "books_clean.csv"
)

transactions = pd.read_csv(
    CLEAN_DIR / "transactions_clean.csv"
)

transactions["transaction_date"] = pd.to_datetime(
    transactions["transaction_date"]
)

# =====================================================
# USER FEATURES
# =====================================================

user_txn_count = (
    transactions.groupby("user_id")
    .size()
    .reset_index(name="user_total_transactions")
)

user_unique_books = (
    transactions.groupby("user_id")["book_id"]
    .nunique()
    .reset_index(name="user_unique_books")
)

user_active_days = (
    transactions.groupby("user_id")["transaction_date"]
    .agg(
        lambda x:
        (x.max() - x.min()).days
    )
    .reset_index(name="user_active_days")
)

user_features = users.merge(
    user_txn_count,
    on="user_id",
    how="left"
)

user_features = user_features.merge(
    user_unique_books,
    on="user_id",
    how="left"
)

user_features = user_features.merge(
    user_active_days,
    on="user_id",
    how="left"
)

user_features.fillna(0, inplace=True)

user_features["user_activity_score"] = (
    user_features["user_total_transactions"]
    * 0.7
    +
    user_features["user_unique_books"]
    * 0.3
)

# =====================================================
# MEMBERSHIP DAYS
# =====================================================

user_features["signup_date"] = pd.to_datetime(
    user_features["signup_date"],
    errors="coerce"
)

today = pd.Timestamp.today()

user_features["membership_days"] = (
    today -
    user_features["signup_date"]
).dt.days

user_features["membership_days"] = (
    user_features["membership_days"]
    .fillna(0)
)

# =====================================================
# BOOK FEATURES
# =====================================================

book_popularity = (
    transactions.groupby("book_id")
    .size()
    .reset_index(name="book_popularity")
)

book_unique_users = (
    transactions.groupby("book_id")["user_id"]
    .nunique()
    .reset_index(name="book_unique_users")
)

books["publication_year"] = pd.to_numeric(
    books["publication_year"],
    errors="coerce"
)

books["book_age"] = (
    pd.Timestamp.today().year
    -
    books["publication_year"]
)

book_features = books.merge(
    book_popularity,
    on="book_id",
    how="left"
)

book_features = book_features.merge(
    book_unique_users,
    on="book_id",
    how="left"
)

book_features.fillna(0, inplace=True)

# =====================================================
# GENRE POPULARITY
# =====================================================

genre_popularity = (
    books["genre"]
    .value_counts()
    .to_dict()
)

book_features["genre_popularity"] = (
    book_features["genre"]
    .map(genre_popularity)
)

# =====================================================
# AUTHOR POPULARITY
# =====================================================

author_popularity = (
    books["author"]
    .value_counts()
    .to_dict()
)

book_features["author_popularity"] = (
    book_features["author"]
    .map(author_popularity)
)

# =====================================================
# INTERACTION FEATURES
# =====================================================

interaction_features = (
    transactions.copy()
)

weight_map = {
    "paperback_purchase": 5,
    "ebook_rental": 3,
    "audiobook_rental": 2
}

interaction_features[
    "interaction_weight"
] = (
    interaction_features["transaction_type"]
    .map(weight_map)
    .fillna(1)
)

latest_date = (
    interaction_features[
        "transaction_date"
    ].max()
)

interaction_features[
    "days_since_interaction"
] = (
    latest_date
    -
    interaction_features[
        "transaction_date"
    ]
).dt.days

# =====================================================
# SAVE
# =====================================================

user_features.to_csv(
    FEATURE_DIR /
    "user_features.csv",
    index=False
)

book_features.to_csv(
    FEATURE_DIR /
    "book_features.csv",
    index=False
)

interaction_features.to_csv(
    FEATURE_DIR /
    "interaction_features.csv",
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\nFeature Engineering Complete")

print(
    "User Features:",
    user_features.shape
)

print(
    "Book Features:",
    book_features.shape
)

print(
    "Interaction Features:",
    interaction_features.shape
)
