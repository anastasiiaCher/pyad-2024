import pickle
import re
import nltk
import pandas as pd
import sklearn

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

nltk.download("stopwords")
nltk.download("punkt")


def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    books = df
    
    # Исправляем строку 209538
    books.loc[209538, "Book-Title"] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
    books.loc[209538, "Book-Author"] = "Michael Teitelbaum"
    books.loc[209538, "Year-Of-Publication"] = "2000"
    books.loc[209538, "Publisher"] = "DK Publishing Inc"
    
    # Исправляем строку 220731
    books.loc[220731, "Book-Title"] = "Peuple du ciel, suivi de 'Les Bergers'"
    books.loc[220731, "Book-Author"] = "Jean-Marie Gustave Le Clézio"
    books.loc[220731, "Year-Of-Publication"] = "2003"
    books.loc[220731, "Publisher"] = "Gallimard"
    
    # Исправляем строку 221678
    books.loc[221678, "Book-Title"] = "DK Readers: Creating the X-Men, How Comic Book Artists Work (Level 4: Proficient Readers)"
    books.loc[221678, "Book-Author"] = "Michael Teitelbaum"
    books.loc[221678, "Year-Of-Publication"] = "2000"
    books.loc[221678, "Publisher"] = "DK Publishing Inc"

    # Получим текущий год
    current_year = 2024
    
    # Фильтрация данных
    books = books[(books["Year-Of-Publication"].astype(str).str.isdigit())]  # Оставляем только числовые значения
    books["Year-Of-Publication"] = books["Year-Of-Publication"].astype(int)  # Приводим к целым числам
    books = books[books["Year-Of-Publication"] <= current_year]  # Удаляем будущие годы
    
    # Проверим результат
    books["Year-Of-Publication"].describe()

    # Удаление пропусков в критически важных столбцах
    books = books.dropna(subset=["Book-Author", "Publisher"])
    
    # Удаление столбцов с картинками
    books = books.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"])

    return books


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv
    Целевой переменной в этой задаче будет средний рейтинг книги,
    поэтому в предобработку (помимо прочего) нужно включить:
    1. Замену оценки книги пользователем на среднюю оценку книги всеми пользователями.
    2. Расчет числа оценок для каждой книги (опционально)."""

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


def title_preprocessing(text: str) -> str:
    """Функция для нормализации текстовых данных в стобце Book-Title:
    - токенизация
    - удаление стоп-слов
    - удаление пунктуации
    Опционально можно убрать шаги или добавить дополнительные.
    """

    pass


def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Бинаризовать или представить в виде чисел категориальные столбцы (кроме названий)
    2. Разбить данные на тренировочную и обучающую выборки
    3. Векторизовать подвыборки и создать датафреймы из векторов (размер вектора названия в тестах – 1000)
    4. Сформировать итоговые X_train, X_test, y_train, y_test
    5. Обучить и протестировать SGDRegressor
    6. Подобрать гиперпараметры (при необходимости)
    7. Сохранить модель"""
    # Получаем средний рейтинг для каждой книги
    book_avg_ratings = ratings.groupby("ISBN")["Book-Rating"].mean()
    
    # Объединяем информацию о книгах с их рейтингами
    merged_books = books[books["ISBN"].isin(book_avg_ratings.index)]
    merged_books["Average-Rating"] = merged_books["ISBN"].map(book_avg_ratings)

    merged_books = merged_books.dropna(subset=["Average-Rating", "Book-Author", "Publisher", "Year-Of-Publication", "Book-Title"])

    # Инициализируем векторизатор для названий книг
    tfidf = TfidfVectorizer(stop_words="english", max_features=500)
    
    # Векторизуем названия книг
    X_title = tfidf.fit_transform(merged_books["Book-Title"])

    # Преобразуем столбцы "Book-Author", "Publisher", "Year-Of-Publication"
    encoder_author = LabelEncoder()
    merged_books["Author-Encoded"] = encoder_author.fit_transform(merged_books["Book-Author"])
    
    encoder_publisher = LabelEncoder()
    merged_books["Publisher-Encoded"] = encoder_publisher.fit_transform(merged_books["Publisher"])
    
    merged_books["Year-Encoded"] = merged_books["Year-Of-Publication"].astype(int)

    # Объединяем все признаки в одну матрицу
    X = sp.hstack([X_title, merged_books[["Author-Encoded", "Publisher-Encoded", "Year-Encoded"]].values])
    
    # Целевая переменная - это средний рейтинг
    y = merged_books["Average-Rating"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.toarray())

    # Инициализация модели
    model = SGDRegressor(max_iter=1000, tol=1e-3)
    
    # Обучение модели
    model.fit(X_scaled, y)
    
    # Предсказания
    y_pred = model.predict(X_scaled)
    
    # Вычисление MAE
    mae = mean_absolute_error(y, y_pred)
    print(f"MAE: {mae}")

    with open("linreg.pkl", "wb") as file:
        pickle.dump(model, file)
