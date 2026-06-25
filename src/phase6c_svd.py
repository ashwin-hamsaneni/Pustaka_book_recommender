import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy

# =====================================================
# PATHS
# =====================================================

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models/svd")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# LOAD
# =====================================================

train_df = pd.read_csv(
    PROCESSED_DIR /
    "train_interactions.csv"
)

test_df = pd.read_csv(
    PROCESSED_DIR /
    "test_interactions.csv"
)

print("Train Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

# =====================================================
# SURPRISE DATASET
# =====================================================

reader = Reader(
    rating_scale=(1, 5)
)

train_data = Dataset.load_from_df(
    train_df[
        [
            "user_id",
            "book_id",
            "interaction_weight"
        ]
    ],
    reader
)

trainset = train_data.build_full_trainset()

# =====================================================
# TRAIN SVD
# =====================================================

model = SVD(
    n_factors=100,
    n_epochs=30,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

print("\nTraining SVD...")

model.fit(trainset)

# =====================================================
# TEST EVALUATION
# =====================================================

testset = list(

    zip(

        test_df["user_id"],

        test_df["book_id"],

        test_df["interaction_weight"]

    )
)

predictions = []

for uid, iid, rating in testset:

    predictions.append(
        model.predict(
            uid,
            iid,
            rating
        )
    )

rmse = accuracy.rmse(
    predictions,
    verbose=False
)

mae = accuracy.mae(
    predictions,
    verbose=False
)

print(f"\nRMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")

# =====================================================
# GENERATE TOP-N RECOMMENDATIONS
# =====================================================

all_books = (
    train_df["book_id"]
    .unique()
)

user_seen = (
    train_df.groupby("user_id")
    ["book_id"]
    .apply(set)
    .to_dict()
)

recommendation_rows = []

users = train_df["user_id"].unique()

print("\nGenerating Recommendations...")

for idx, user in enumerate(users):

    seen_books = user_seen.get(
        user,
        set()
    )

    candidate_books = [

        book
        for book in all_books
        if book not in seen_books

    ]

    predictions_user = []

    for book in candidate_books:

        pred = model.predict(
            user,
            book
        )

        predictions_user.append(

            (
                book,
                pred.est
            )

        )

    predictions_user.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top10 = predictions_user[:10]

    for rank, (book, score) in enumerate(
        top10,
        start=1
    ):

        recommendation_rows.append({

            "user_id": user,

            "book_id": book,

            "predicted_score": score,

            "rank": rank

        })

    if idx % 100 == 0:

        print(
            f"Processed {idx}/{len(users)} users"
        )

recommendations = pd.DataFrame(
    recommendation_rows
)

# =====================================================
# SAVE
# =====================================================

recommendations.to_csv(

    MODEL_DIR /
    "user_recommendations.csv",

    index=False

)

metrics = pd.DataFrame({

    "Metric": [

        "RMSE",
        "MAE",

        "Users",
        "Recommendations"

    ],

    "Value": [

        rmse,
        mae,

        len(users),
        len(recommendations)

    ]

})

metrics.to_csv(

    MODEL_DIR /
    "svd_metrics.csv",

    index=False

)

joblib.dump(

    model,

    MODEL_DIR /
    "svd_model.pkl"

)

# =====================================================
# SUMMARY
# =====================================================

print("\nSVD COMPLETE")

print(
    f"Users: {len(users)}"
)

print(
    f"Recommendations: {len(recommendations)}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"MAE: {mae:.4f}"
)

print("\nSaved Files:")

print(
    "models/svd/svd_model.pkl"
)

print(
    "models/svd/user_recommendations.csv"
)

print(
    "models/svd/svd_metrics.csv"
)
