import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PATHS
# =====================================================

FEATURE_DIR = Path("data/features")

MODEL_DIR = Path("models/content_based")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# LOAD DATA
# =====================================================

books = pd.read_csv(
    FEATURE_DIR / "book_features.csv"
)

print("Books Loaded")
print(books.shape)

# =====================================================
# CLEAN TEXT COLUMNS
# =====================================================

text_columns = [
    "genre",
    "author",
    "language",
    "format"
]

for col in text_columns:

    books[col] = (
        books[col]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )

# =====================================================
# CREATE CONTENT PROFILE
# =====================================================

books["content_profile"] = (

    books["genre"] + " " +

    books["author"] + " " +

    books["language"] + " " +

    books["format"]

)

# =====================================================
# TF-IDF
# =====================================================

vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(
    books["content_profile"]
)

print(
    "TF-IDF Shape:",
    tfidf_matrix.shape
)

# =====================================================
# COSINE SIMILARITY
# =====================================================

similarity_matrix = cosine_similarity(
    tfidf_matrix
)

print(
    "Similarity Matrix Shape:",
    similarity_matrix.shape
)

# =====================================================
# BOOK INDEX MAPPING
# =====================================================

book_index = pd.Series(
    books.index,
    index=books["book_id"]
)

# =====================================================
# RECOMMENDATION FUNCTION
# =====================================================

def recommend_books(
    book_id,
    top_n=10
):

    if book_id not in book_index:

        return []

    idx = book_index[book_id]

    scores = list(
        enumerate(
            similarity_matrix[idx]
        )
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = scores[1:top_n+1]

    recommended_indices = [
        i[0]
        for i in scores
    ]

    return books.iloc[
        recommended_indices
    ][
        [
            "book_id",
            "title",
            "author",
            "genre"
        ]
    ]

# =====================================================
# GENERATE ALL RECOMMENDATIONS
# =====================================================

recommendation_rows = []

for book_id in books["book_id"]:

    recommendations = recommend_books(
        book_id,
        top_n=10
    )

    for rank, row in enumerate(
        recommendations.itertuples(),
        start=1
    ):

        recommendation_rows.append({

            "source_book": book_id,

            "recommended_book": row.book_id,

            "rank": rank,

            "title": row.title,

            "author": row.author,

            "genre": row.genre

        })

recommendations_df = pd.DataFrame(
    recommendation_rows
)

# =====================================================
# SAVE
# =====================================================

recommendations_df.to_csv(
    MODEL_DIR /
    "content_recommendations.csv",
    index=False
)

joblib.dump(
    vectorizer,
    MODEL_DIR /
    "tfidf_vectorizer.pkl"
)

joblib.dump(
    similarity_matrix,
    MODEL_DIR /
    "content_similarity.pkl"
)

# =====================================================
# SAMPLE OUTPUT
# =====================================================

sample_book = books.iloc[0]["book_id"]

print("\nSample Book")
print(sample_book)

print(
    recommend_books(
        sample_book,
        5
    )
)

print("\nSaved Files")

print(
    "models/content_based/content_recommendations.csv"
)

print(
    "models/content_based/tfidf_vectorizer.pkl"
)

print(
    "models/content_based/content_similarity.pkl"
)

print("\nContent-Based Model Complete")
