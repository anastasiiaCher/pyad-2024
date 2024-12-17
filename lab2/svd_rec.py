import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv"""
    df = df[df["Book-Rating"] > 0]

    book_counts = df["ISBN"].value_counts()
    df = df[df["ISBN"].isin(book_counts[book_counts > 1].index)]

    user_counts = df["User-ID"].value_counts()
    df = df[df["User-ID"].isin(user_counts[user_counts > 1].index)]

    return df


def modeling(ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Разбить данные на тренировочную и обучающую выборки
    2. Обучить и протестировать SVD
    3. Подобрать гиперпараметры (при необходимости)
    4. Сохранить модель"""
    ratings = ratings_preprocessing(ratings)

    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[["User-ID", "ISBN", "Book-Rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=108)

    test_data = [(uid, iid, r) for (uid, iid, r) in testset]
    test_ratings = pd.DataFrame(test_data, columns=["User-ID", "ISBN", "Book-Rating"])
    test_ratings.to_csv("svd_test.csv", index=False)

    param_grid = {
        'n_factors': [50, 100, 150],
        'n_epochs': [20, 30, 40],
        'lr_all': [0.002, 0.005, 0.01],
        'reg_all': [0.02, 0.1, 0.2]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['mae'], cv=3)
    gs.fit(data)

    svd = gs.best_estimator['mae']
    print("Best MAE parameters:", gs.best_params['mae'])

    svd.fit(trainset)

    # Tests
    predictions = svd.test(testset)
    mae = accuracy.mae(predictions)
    print("MAE:", mae)

    with open("svd.pkl", "wb") as file:
        pickle.dump(svd, file)



if __name__ == "__main__":
    ratings = pd.read_csv("Ratings.csv")
    modeling(ratings)
