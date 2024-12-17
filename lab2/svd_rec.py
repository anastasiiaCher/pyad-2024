import pickle

import pandas
import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split, GridSearchCV


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv"""
    df = df[df["Book-Rating"] > 0].copy()

    book_counts = df["ISBN"].value_counts()
    valid_books = book_counts[book_counts > 1].index
    df = df[df["ISBN"].isin(valid_books)]

    user_counts = df["User-ID"].value_counts()
    valid_users = user_counts[user_counts > 1].index
    df = df[df["User-ID"].isin(valid_users)]
    return df


def modeling(ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Разбить данные на тренировочную и обучающую выборки
    2. Обучить и протестировать SVD
    3. Подобрать гиперпараметры (при необходимости)
    4. Сохранить модель"""
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
    train_set = data.build_full_trainset()
    param_grid = {
        'n_factors': [50, 100],
        'n_epochs': [20, 30],
        'lr_all': [0.005, 0.01],
        'reg_all': [0.02, 0.1]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['mae'], cv=3)
    gs.fit(data)
    print(gs.best_params['mae'])
    svd = gs.best_estimator['mae']
    svd.fit(train_set)
    train_set, test_set = train_test_split(data, test_size=0.1)
    predictions = svd.test(test_set)
    mae_svd = accuracy.mae(predictions)
    print(f"MAE для SVD модели: {mae_svd}")
    with open('svd.pkl', 'wb') as f:
        pickle.dump(svd, f)


ratings = pandas.read_csv("Ratings.csv")
ratings_cleaned = ratings_preprocessing(ratings)
modeling(ratings_cleaned)
