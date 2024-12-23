import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv
    Целевой переменной в этой задаче будет средний рейтинг книги,
    поэтому в предобработку (помимо прочего) нужно включить:
    1. Замену оценки книги пользователем на среднюю оценку книги всеми пользователями.
    2. Расчет числа оценок для каждой книги (опционально)."""
    ratings = df

    # Удаляем нулевые рейтинги
    ratings = ratings[ratings["Book-Rating"].astype(int) != 0]

    # Удаляем книги с единственной оценкой
    book_counts = ratings['ISBN'].value_counts()
    books_to_keep = book_counts[book_counts > 1].index
    ratings = ratings[ratings['ISBN'].isin(books_to_keep)]

    # Удаляем пользователей с единственной оценкой
    user_counts = ratings['User-ID'].value_counts()
    users_to_keep = user_counts[user_counts > 1].index
    ratings = ratings[ratings['User-ID'].isin(users_to_keep)]


    return ratings


def modeling(ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Разбить данные на тренировочную и обучающую выборки
    2. Обучить и протестировать SVD
    3. Подобрать гиперпараметры (при необходимости)
    4. Сохранить модель"""

    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import train_test_split
    from surprise import accuracy
    import pickle

    ratings = ratings_preprocessing(ratings)

    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[["User-ID", "ISBN", "Book-Rating"]], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD()
    model.fit(trainset)

    predictions = model.test(testset)
    mae = accuracy.mae(predictions)

    # Проверка MAE
    if mae < 1.3:
        with open("svd.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"Модель сохранена. MAE: {mae}")
    else:
        print(f"MAE выше порога: {mae}")
