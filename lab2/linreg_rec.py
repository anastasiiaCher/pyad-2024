import pickle
import re
from datetime import datetime

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
    expected_columns = books.columns
    books = books[~books["Book-Author"].str.contains(r"\d", na=False)]

    # 2. Удаление записей с будущими годами
    current_year = datetime.now().year

    # Преобразование столбца Year-Of-Publication в числовой формат
    books["Year-Of-Publication"] = pd.to_numeric(
        books["Year-Of-Publication"], errors="coerce"
    )

    # Удаляем строки с NaN в Year-Of-Publication после преобразования
    books = books.dropna(subset=["Year-Of-Publication"])

    # Преобразуем Year-Of-Publication обратно в int (после фильтрации NaN)
    books["Year-Of-Publication"] = books["Year-Of-Publication"].astype(int)

    # Удаляем строки, где год больше текущего
    books = books[books["Year-Of-Publication"] <= current_year]

    # 3. Заполнение пропусков
    books.fillna(
        {
            "Book-Title": "Unknown",
            "Book-Author": "Unknown",
            "Year-Of-Publication": books["Year-Of-Publication"].mean().astype(int),
            "Publisher": "Unknown",
        },
        inplace=True,
    )

    # 4. Удаление столбцов с URL
    books = books.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"])

    return books


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обрабатывает таблицу с рейтингами:
    1. Исключает записи с рейтингом 0.
    2. Исключает книги с единственной оценкой.
    3. Исключает пользователей, которые оценили только одну книгу.
    """

    # Исключаем записи с рейтингом 0
    df = df[df["Book-Rating"] > 0]

    # Считаем количество оценок для каждой книги
    book_rating_counts = df["ISBN"].value_counts()

    # Оставляем только книги, которые оценивались более одного раза
    books_with_multiple_ratings = book_rating_counts[book_rating_counts > 1].index
    df = df[df["ISBN"].isin(books_with_multiple_ratings)]

    # Считаем количество оценок от каждого пользователя
    user_rating_counts = df["User-ID"].value_counts()

    # Оставляем только пользователей, которые оценили более одной книги
    users_with_multiple_ratings = user_rating_counts[user_rating_counts > 1].index
    df = df[df["User-ID"].isin(users_with_multiple_ratings)]

    # Группируем данные по ISBN и вычисляем средний рейтинг
    result = (
        df.groupby("ISBN")
        .agg(
            Book_Rating=("Book-Rating", "mean"),  # Средний рейтинг книги
        )
        .reset_index()
    )

    return result


def title_preprocessing(text: str) -> str:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string

    # Токенизация
    tokens = word_tokenize(text.lower())

    # Удаление стоп-слов
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Удаление пунктуации
    tokens = [word for word in tokens if word not in string.punctuation]

    # Возвращаем строку без стоп-слов и пунктуации
    return " ".join(tokens)


def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Бинаризовать или представить в виде чисел категориальные столбцы (кроме названий)
    2. Разбить данные на тренировочную и обучающую выборки
    3. Векторизовать подвыборки и создать датафреймы из векторов (размер вектора названия в тестах – 1000)
    4. Сформировать итоговые X_train, X_test, y_train, y_test
    5. Обучить и протестировать SGDRegressor
    6. Подобрать гиперпараметры (при необходимости)
    7. Сохранить модель"""

    # Предобработка данных
    books = books_preprocessing(books)
    ratings = ratings_preprocessing(ratings)

    # Объединяем данные книг и рейтингов
    data = pd.merge(ratings, books, on="ISBN", how="inner")

    # Удаляем все строки с NaN в любых столбцах
    data = data.dropna()

    # 1. Преобразуем категориальные переменные в числовые
    le_publisher = LabelEncoder()
    le_author = LabelEncoder()

    data["Publisher"] = le_publisher.fit_transform(data["Publisher"])
    data["Book-Author"] = le_author.fit_transform(data["Book-Author"])
    data["Book-Title"] = data["Book-Title"].apply(title_preprocessing)
    # 2. Векторизация названий книг
    tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
    book_titles = tfidf.fit_transform(data["Book-Title"]).toarray()
    with open("le_publisher.pkl", "wb") as file:
        pickle.dump(le_publisher, file)
    with open("le_author.pkl", "wb") as file:
        pickle.dump(le_author, file)
    with open("tfidf.pkl", "wb") as file:
        pickle.dump(tfidf, file)
    scaler = StandardScaler()
    books_scaled = scaler.fit_transform(
        data[["Book-Author", "Publisher", "Year-Of-Publication"]]
    )
    X = pd.concat([pd.DataFrame(books_scaled), pd.DataFrame(book_titles)], axis=1)
    # 3. Разбиение данных на тренировочную и тестовую выборки
    print(X.shape)
    y = data["Book_Rating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # 5. Обучение модели
    model = SGDRegressor(max_iter=5000, tol=1e-5, learning_rate="adaptive")
    model.fit(X_train, y_train)

    # 6. Предсказания и оценка модели
    y_pred = model.predict(X_test)
    mae_score = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae_score:.4f}")

    # 7. Сохранение модели
    with open("linreg.pkl", "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    import nltk

    nltk.download("punkt_tab")
    modeling(pd.read_csv("Books.csv"), pd.read_csv("Ratings.csv"))
