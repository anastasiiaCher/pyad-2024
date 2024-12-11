import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv"""

    df = df[df['Book-Rating'] > 0]

    book_counts = df['ISBN'].value_counts()
    valid_books = book_counts[book_counts > 5].index
    df = df[df['ISBN'].isin(valid_books)]

    user_counts = df['User-ID'].value_counts()
    valid_users = user_counts[user_counts > 5].index
    df = df[df['User-ID'].isin(valid_users)]
    df = df.drop_duplicates()
    print(df)
    return df
    pass


def modeling(ratings: pd.DataFrame) -> None:
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

    test_data = pd.read_csv("svd_test.csv")
    testset = list(zip(test_data['User-ID'], test_data['ISBN'], test_data['Book-Rating']))
    trainset = data.build_full_trainset()

    svd = SVD(n_factors=50, lr_all=0.005, reg_all=0.02, n_epochs=30, random_state=42)
    svd.fit(trainset)

    predictions = svd.test(testset)

    mae = accuracy.mae(predictions)

    if mae < 1.5:
        with open("svd.pkl", "wb") as file:
            pickle.dump(svd, file)
        print(f"Model saved successfully with MAE: {mae}")
    else:
        print(f"Model not saved. MAE is too high: {mae}")
def main():
    ratings = pd.read_csv("Ratings.csv")
    print(ratings.head())
    ratings = ratings_preprocessing(ratings)
    modeling(ratings)

if __name__ == "__main__":
    main()