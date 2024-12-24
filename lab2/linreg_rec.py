import pickle
import re
import string
import nltk
import pandas as pd
import sklearn
import numpy as np


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

nltk.download("stopwords")
nltk.download("punkt")


def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Books.scv"""
    books = df[df['Year-Of-Publication'].map(str).str.isnumeric()]

    # Преобразование 'year' в числовой формат и удаление записей, где год > 2024
    books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
    books = books[books['Year-Of-Publication'] <= 2024]
    
    # Удаление столбцов с ссылками на обложки
    columns_to_drop = ['Image-URL-S', 'Image-URL-M', 'Image-URL-L']
    books = books.drop(columns=columns_to_drop)

    return books


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:    
    """Функция для предобработки таблицы Ratings.scv"""

    # Не будем использовать для обучения записи с рейтингом равным 0:
    df_raitings = df[df['Book-Rating'].astype(int) != 0]

    # Будем использовать для обучения алгоритмов те книги, которым оценка поставлена больше min_count раз:
    isbn_to_count = df_raitings['ISBN'].value_counts()
    books_for_processing = isbn_to_count[isbn_to_count > 1].index
    df_raitings = df_raitings[df_raitings['ISBN'].isin(books_for_processing)]

    # Будем использовать для обучения алгоритмов пользователей, которые поставили оценку больше min_count раз:
    user_to_count = df_raitings['User-ID'].value_counts()
    users_for_processing = user_to_count[user_to_count > 1].index
    df_raitings = df_raitings[df_raitings['User-ID'].isin(users_for_processing)]
    
    # Замена оценки книги пользователем на среднюю оценку книги всеми пользователями:
    average_ratings = df_raitings.groupby('ISBN')['Book-Rating'].mean().reset_index()
    average_ratings.rename(columns={'Book-Rating': 'Average-Book-Rating'}, inplace=True)

    return average_ratings


def title_preprocessing(text: str) -> str:
    """Функция для нормализации текстовых данных в стобце Book-Title:
    - токенизация
    - удаление стоп-слов
    - удаление пунктуации
    Опционально можно убрать шаги или добавить дополнительные.
    """
    
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in string.punctuation]
    return " ".join(tokens)

def prepare_data(books: pd.DataFrame, ratings: pd.DataFrame):
    # Объединение таблиц
    df = pd.merge(ratings, books, on="ISBN", how="inner")

    # Преобразуем категориальные переменные в числовые, используя LabelEncoder
    le_publisher = LabelEncoder()
    le_author = LabelEncoder()

    df["Publisher"] = le_publisher.fit_transform(df["Publisher"])
    df["Book-Author"] = le_author.fit_transform(df["Book-Author"])
    df["Book-Title"] = df["Book-Title"].apply(title_preprocessing)
    
    # Векторизация названий книг
    tfidf = TfidfVectorizer(max_features=1001, stop_words="english")
    book_titles = tfidf.fit_transform(df["Book-Title"]).toarray()

    scaler = StandardScaler()
    books_scaled = scaler.fit_transform(
        df[["Book-Author", "Publisher", "Year-Of-Publication"]]
    )
    X = pd.concat([pd.DataFrame(books_scaled), pd.DataFrame(book_titles)], axis=1)
    y = df["Average-Book-Rating"]
    return X, y, pd.DataFrame(df["ISBN"])
    

def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Бинаризовать или представить в виде чисел категориальные столбцы (кроме названий)
    2. Разбить данные на тренировочную и обучающую выборки
    3. Векторизовать подвыборки и создать датафреймы из векторов (размер вектора названия в тестах – 1000)
    4. Сформировать итоговые X_train, X_test, y_train, y_test
    5. Обучить и протестировать SGDRegressor
    6. Подобрать гиперпараметры (при необходимости)
    7. Сохранить модель"""
    
    X, y, df_isbn = prepare_data(books, ratings)

    # Формирование итоговых X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Обучение и тестирование SGDRegressor
    model = SGDRegressor(max_iter=5000, tol=1e-5, learning_rate="adaptive")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae:.4f}")

    # Сохранение модели
    if mae < 1.5:
        with open("linreg.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"Модель сохранена. MAE: {mae}")
    else:
        print(f"Модель не сохранена. MAE: {mae}")