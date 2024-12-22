import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    ratings = df.rename(columns={'Book-Rating': 'Rating'})
    ratings = ratings.query("Rating != 0")
    min_ratings = 2
    book_counts = ratings.groupby('ISBN')['User-ID'].nunique()
    user_counts = ratings.groupby('User-ID')['ISBN'].nunique()
    good_books = book_counts[book_counts >= min_ratings].index
    good_users = user_counts[user_counts >= min_ratings].index
    filtered1_ratings = ratings[(ratings['ISBN'].isin(good_books)) & (ratings['User-ID'].isin(good_users))]
    return filtered1_ratings


def modeling(ratings: pd.DataFrame) -> None:
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Rating']], reader)
    train_set, test_set = train_test_split(data, test_size=0.3)
    svd = SVD(n_factors=128, n_epochs=15,  verbose=True)
    svd.fit(train_set)
    with open("svd.pkl", "wb") as file:
        pickle.dump(svd, file)


test = pd.read_csv("Ratings.csv")
test_1 = ratings_preprocessing(test)
print(test_1)
modeling(test_1)
