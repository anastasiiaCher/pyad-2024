import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv"""
    ratings = df
    # Считаем количество оценок для каждой книги (ISBN)
    book_rating_counts = ratings["ISBN"].value_counts()
    
    # Оставляем только те ISBN, у которых больше одной оценки
    valid_books = book_rating_counts[book_rating_counts > 1].index
    
    # Фильтруем исходную таблицу ratings, оставляя только книги из filtered_books
    filtered_ratings = ratings[ratings["ISBN"].isin(filtered_books["ISBN"])]
    
    # Считаем количество оценок для каждого пользователя
    user_rating_counts = filtered_ratings["User-ID"].value_counts()
    
    # Получаем список пользователей, которые оценили хотя бы 2 книги
    valid_users = user_rating_counts[user_rating_counts > 1].index
    
    # Фильтруем таблицу ratings
    filtered_ratings = filtered_ratings[filtered_ratings["User-ID"].isin(valid_users)]

    filtered_ratings = filtered_ratings[filtered_ratings["Book-Rating"] > 2]
    
    return filtered_ratings


def modeling(ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Разбить данные на тренировочную и обучающую выборки
    2. Обучить и протестировать SVD
    3. Подобрать гиперпараметры (при необходимости)
    4. Сохранить модель"""

    # Создаем объект Reader (указываем диапазон рейтингов)
    reader = Reader(rating_scale=(1, 10))  # Диапазон значений рейтингов
    
    # Создаем surprise Dataset из pandas DataFrame
    data = Dataset.load_from_df(ratings[["User-ID", "ISBN", "Book-Rating"]], reader)
    
    # Разделяем данные на trainset и testset (25% - тестовые)
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    
    # Создаем и обучаем алгоритм SVD
    algo = SVD(reg_bu=0.02, reg_bi=0.02, reg_pu=0.02, reg_qi=0.02, n_epochs=30)
    algo.fit(trainset)
    
    # Делаем предсказания на тестовом наборе
    predictions = algo.test(testset)
    
    # Оцениваем качество модели
    mae = accuracy.mae(predictions)  # Mean Absolute Error
    rmse = accuracy.rmse(predictions)  # Root Mean Square Error
    
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    # ...
    svd = SVD()
    # ...
    with open("svd.pkl", "wb") as file:
        pickle.dump(algo, file)
