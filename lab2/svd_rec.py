import joblib
import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.accuracy import mae
from surprise.model_selection import train_test_split


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Book-Rating"] > 0].copy()

    book_counts = df["ISBN"].value_counts()
    valid_books = book_counts[book_counts > 1].index
    df = df[df["ISBN"].isin(valid_books)]

    user_counts = df["User-ID"].value_counts()
    valid_users = user_counts[user_counts > 1].index
    df = df[df["User-ID"].isin(valid_users)]

    return df


def modeling(ratings: pd.DataFrame) -> None:
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[["User-ID", "ISBN", "Book-Rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    svd = SVD(random_state=42)
    svd.fit(trainset)

    predictions = svd.test(testset)
    score = mae(predictions)

    print(score)

    with open("svd.pkl", "wb") as file:
        pickle.dump(svd, file)
