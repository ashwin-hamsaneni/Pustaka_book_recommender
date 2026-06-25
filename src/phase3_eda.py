import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# =====================================================
# PATHS
# =====================================================

CLEAN_DIR = Path("data/cleaned")
EDA_DIR = Path("reports/eda")

EDA_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD DATA
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

print("Datasets Loaded")

print(users.shape)
print(books.shape)
print(transactions.shape)

# =====================================================
# STYLE
# =====================================================

plt.style.use("default")
sns.set_theme()

# =====================================================
# 1. USER AGE DISTRIBUTION
# =====================================================

plt.figure(figsize=(10,6))

sns.histplot(
    users["age"],
    bins=20,
    kde=True
)

plt.title("User Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    EDA_DIR / "01_user_age_distribution.png"
)

plt.close()

# =====================================================
# 2. GENDER DISTRIBUTION
# =====================================================

plt.figure(figsize=(8,6))

users["gender"].value_counts().plot(
    kind="bar"
)

plt.title("Gender Distribution")

plt.tight_layout()

plt.savefig(
    EDA_DIR / "02_gender_distribution.png"
)

plt.close()

# =====================================================
# 3. SUBSCRIPTION DISTRIBUTION
# =====================================================

plt.figure(figsize=(12,6))

users["subscription_type"].value_counts().head(10).plot(
    kind="bar"
)

plt.title("Subscription Type Distribution")

plt.tight_layout()

plt.savefig(
    EDA_DIR / "03_subscription_distribution.png"
)

plt.close()

# =====================================================
# 4. GENRE POPULARITY
# =====================================================

plt.figure(figsize=(12,6))

books["genre"].value_counts().head(15).plot(
    kind="bar"
)

plt.title("Top Genres")

plt.tight_layout()

plt.savefig(
    EDA_DIR / "04_genre_popularity.png"
)

plt.close()

# =====================================================
# 5. LANGUAGE DISTRIBUTION
# =====================================================

plt.figure(figsize=(10,6))

books["language"].value_counts().plot(
    kind="bar"
)

plt.title("Book Language Distribution")

plt.tight_layout()

plt.savefig(
    EDA_DIR / "05_language_distribution.png"
)

plt.close()

# =====================================================
# 6. TRANSACTION TYPES
# =====================================================

plt.figure(figsize=(10,6))

transactions["transaction_type"].value_counts().plot(
    kind="bar"
)

plt.title("Transaction Types")

plt.tight_layout()

plt.savefig(
    EDA_DIR / "06_transaction_types.png"
)

plt.close()

# =====================================================
# 7. BOOK POPULARITY
# =====================================================

book_popularity = (
    transactions["book_id"]
    .value_counts()
    .head(20)
)

plt.figure(figsize=(12,6))

book_popularity.plot(
    kind="bar"
)

plt.title("Top 20 Most Popular Books")

plt.tight_layout()

plt.savefig(
    EDA_DIR / "07_book_popularity.png"
)

plt.close()

# =====================================================
# 8. USER ACTIVITY
# =====================================================

user_activity = (
    transactions["user_id"]
    .value_counts()
)

plt.figure(figsize=(10,6))

sns.histplot(
    user_activity,
    bins=30
)

plt.title("User Activity Distribution")

plt.xlabel("Transactions per User")

plt.tight_layout()

plt.savefig(
    EDA_DIR / "08_user_activity_distribution.png"
)

plt.close()

# =====================================================
# 9. SPARSITY ANALYSIS
# =====================================================

n_users = users["user_id"].nunique()
n_books = books["book_id"].nunique()

possible_interactions = (
    n_users * n_books
)

actual_interactions = len(transactions)

density = (
    actual_interactions /
    possible_interactions
)

sparsity = (
    1 - density
)

with open(
    EDA_DIR / "09_sparsity_report.txt",
    "w"
) as f:

    f.write(
        f"Users: {n_users}\n"
    )

    f.write(
        f"Books: {n_books}\n"
    )

    f.write(
        f"Actual Interactions: {actual_interactions}\n"
    )

    f.write(
        f"Possible Interactions: {possible_interactions}\n"
    )

    f.write(
        f"Density: {density:.6f}\n"
    )

    f.write(
        f"Sparsity: {sparsity:.6f}\n"
    )

# =====================================================
# 10. COLD START ANALYSIS
# =====================================================

active_users = set(
    transactions["user_id"]
)

cold_users = (
    set(users["user_id"])
    - active_users
)

active_books = set(
    transactions["book_id"]
)

cold_books = (
    set(books["book_id"])
    - active_books
)

with open(
    EDA_DIR / "10_cold_start_report.txt",
    "w"
) as f:

    f.write(
        f"Cold Users: {len(cold_users)}\n"
    )

    f.write(
        f"Cold Books: {len(cold_books)}\n"
    )

# =====================================================
# EDA SUMMARY
# =====================================================

summary = pd.DataFrame({

    "Metric": [

        "Users",
        "Books",
        "Transactions",

        "Interaction Density",
        "Interaction Sparsity",

        "Cold Users",
        "Cold Books"

    ],

    "Value": [

        n_users,
        n_books,
        actual_interactions,

        density,
        sparsity,

        len(cold_users),
        len(cold_books)

    ]
})

summary.to_csv(
    EDA_DIR / "eda_summary.csv",
    index=False
)

print("\nEDA COMPLETE")

print(f"Users: {n_users}")
print(f"Books: {n_books}")
print(f"Transactions: {actual_interactions}")

print(f"Density: {density:.6f}")
print(f"Sparsity: {sparsity:.6f}")

print(f"Cold Users: {len(cold_users)}")
print(f"Cold Books: {len(cold_books)}")

print("\nSaved to reports/eda/")
