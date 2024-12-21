import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv"""

    ratings = pd.read_csv("Ratings.csv")

    ratings = ratings[ratings["Book-Rating"].astype(int) != 0]

    book_counts = ratings["ISBN"].value_counts()
    books_to_keep = book_counts[book_counts > 1].index
    ratings = ratings[ratings["ISBN"].isin(books_to_keep)]

    user_counts = ratings["User-ID"].value_counts()
    users_to_keep = user_counts[user_counts > 1].index
    ratings = ratings[ratings["User-ID"].isin(users_to_keep)]
    return ratings
    pass


def modeling(ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Разбить данные на тренировочную и обучающую выборки
    2. Обучить и протестировать SVD
    3. Подобрать гиперпараметры (при необходимости)
    4. Сохранить модель"""

    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import train_test_split
    from surprise.accuracy import mae

    # Оставляем только необходимые столбцы
    ratings = ratings[["User-ID", "ISBN", "Book-Rating"]]

    # Используем Surprise Reader для загрузки данных
    reader = Reader(
        rating_scale=(1, 10)
    )  # Предполагается, что рейтинги в диапазоне 1-10
    data = Dataset.load_from_df(ratings, reader)

    # Разделяем данные на обучающую и тестовую выборки
    trainset, testset = train_test_split(data, test_size=0.1)
    # from surprise.model_selection import GridSearchCV
    #
    # param_grid = {
    #    "n_factors": [50, 100],
    #    "n_epochs": [20, 30],
    #    "lr_all": [0.005, 0.01],
    #    "reg_all": [0.02, 0.1],
    # }
    #
    # grid_search = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)
    # grid_search.fit(data)
    #
    ## Выводим лучшие параметры
    # print(grid_search.best_params["mae"])
    # Обучение модели SVD
    model = SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.1)
    model.fit(trainset)
    # Оценка модели на тестовом наборе
    predictions = model.test(testset)

    # Подсчет MAE
    model_mae = mae(predictions)
    print(f"MAE модели: {model_mae:.4f}")

    # Проверяем, чтобы MAE был ниже 1.3
    if model_mae <= 1.3:
        print("Модель прошла тест с MAE ниже 1.3!")
    else:
        print("Модель не прошла тест, требуется улучшение.")

    with open("svd.pkl", "wb") as file:
        pickle.dump(model, file)


modeling(ratings_preprocessing(pd.read_csv("Ratings.csv")))
