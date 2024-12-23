import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv"""
    df = df.copy()

    # Удаление нулевых рейтингов
    df = df[df['Book-Rating'] != 0]
    # Задаем минимально кол-во
    min_books_ratings = 2
    
    # Рассчитываем средний рейтинг для каждой книги
    book_avg_ratings = df.groupby('ISBN')['Book-Rating'].mean()
    
    # Заменяем индивидуальные рейтинги на средние для книги
    df['Book-Rating'] = df['ISBN'].map(book_avg_ratings)
    
    # Рассчитываем число оценок для каждой книги
    book_ratings_count = df.groupby('ISBN')['Book-Rating'].count()
    
    # Добавляем количество оценок к исходным данным
    df['Book-Ratings-Count'] = df['ISBN'].map(book_ratings_count)
    
    # Фильтруем книги, у которых меньше min_books_ratings оценок
    df = df[df['Book-Ratings-Count'] >= min_books_ratings]
    
    # Удалим пользователей, которые поставили меньше min_books_ratings оценок
    user_ratings_count = df.groupby('User-ID')['Book-Rating'].count()
    good_users = user_ratings_count[user_ratings_count >= min_books_ratings].index
    df = df[df['User-ID'].isin(good_users)]
    
    return df


def modeling(ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Разбить данные на тренировочную и обучающую выборки
    2. Обучить и протестировать SVD
    3. Подобрать гиперпараметры (при необходимости)
    4. Сохранить модель"""
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Rating']], reader)
    train_set, test_set = train_test_split(data, test_size=0.3)
    svd = SVD(n_factors=128, n_epochs=15, verbose=True)
    svd.fit(train_set)
    with open("svd.pkl", "wb") as file:
        pickle.dump(svd, file)
