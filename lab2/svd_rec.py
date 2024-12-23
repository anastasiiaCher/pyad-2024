import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


def ratings_preprocessing(ratings: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv"""
    ratings = ratings[ratings['Book-Rating'] != 0]

    ratings_counts = ratings['ISBN'].value_counts()
    ratings = ratings[ratings['ISBN'].isin(ratings_counts[ratings_counts > 1].index)]

    user_counts = ratings['User-ID'].value_counts()
    ratings = ratings[ratings['User-ID'].isin(user_counts[user_counts > 1].index)]
    return ratings


def modeling(ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Разбить данные на тренировочную и обучающую выборки
    2. Обучить и протестировать SVD
    3. Подобрать гиперпараметры (при необходимости)
    4. Сохранить модель"""
    reader = Reader(rating_scale=(1, 10))

    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    svd = SVD(n_factors = 50, n_epochs = 30, lr_all = 0.005, reg_all = 0.1) 
    svd.fit(trainset)

    predictions = svd.test(testset)
    mae = accuracy.mae(predictions) 
    print(f"MAE модели: {mae}")

    with open('svd.pkl', 'wb') as file:
        pickle.dump(svd, file)

# test = pd.read_csv("Ratings.csv")
# test_1 = ratings_preprocessing(test)
# modeling(test_1)

