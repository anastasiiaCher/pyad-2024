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
from sklearn.preprocessing import StandardScaler

nltk.download("stopwords")
nltk.download("punkt")


def books_preprocessing(books: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Books.scv"""
    columns_to_drop = ['Image-URL-S',	'Image-URL-M',	'Image-URL-L']
    books = books.drop(columns=columns_to_drop)

    books = books[~(books["Book-Author"].isnull())]
    books = books[~(books["Publisher"].isnull())]
    books["Year-Of-Publication"] = pd.to_numeric(books["Year-Of-Publication"], errors='coerce')
    books = books[~(books["Year-Of-Publication"].isnull())]
    books = books[(books["Year-Of-Publication"] <= 2024)]

    return books


def ratings_preprocessing(ratings: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv
    Целевой переменной в этой задаче будет средний рейтинг книги,
    поэтому в предобработку (помимо прочего) нужно включить:
    1. Замену оценки книги пользователем на среднюю оценку книги всеми пользователями.
    2. Расчет числа оценок для каждой книги (опционально)."""

    ratings = ratings[ratings['Book-Rating'] != 0]

    ratings_counts = ratings['ISBN'].value_counts()
    ratings = ratings[ratings['ISBN'].isin(ratings_counts[ratings_counts > 1].index)]

    user_counts = ratings['User-ID'].value_counts()
    ratings = ratings[ratings['User-ID'].isin(user_counts[user_counts > 1].index)]

    ratings = ratings.groupby('ISBN').agg({'Book-Rating': ['mean', 'count']}).reset_index()
    ratings.columns = ['ISBN', 'Average-Rating', 'Rating-Count']

    return ratings


def title_preprocessing(text: str) -> str:
    """Функция для нормализации текстовых данных в стобце Book-Title:
    - токенизация
    - удаление стоп-слов
    - удаление пунктуации
    Опционально можно убрать шаги или добавить дополнительные.
    """
    tokens = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
    
    return ' '.join(tokens)


def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Бинаризовать или представить в виде чисел категориальные столбцы (кроме названий)
    2. Разбить данные на тренировочную и обучающую выборки
    3. Векторизовать подвыборки и создать датафреймы из векторов (размер вектора названия в тестах – 1000)
    4. Сформировать итоговые X_train, X_test, y_train, y_test
    5. Обучить и протестировать SGDRegressor
    6. Подобрать гиперпараметры (при необходимости)
    7. Сохранить модель"""

    merged_data = books.merge(ratings, on='ISBN', how='inner')

    merged_data['Processed_Title'] = merged_data['Book-Title'].map(title_preprocessing)

    vectorizer = TfidfVectorizer(max_features=1000)
    title_vectors = vectorizer.fit_transform(merged_data['Processed_Title']).toarray()

    category_cols = ['Book-Author', 'Publisher', 'Year-Of-Publication']
    encoded_categories = merged_data[category_cols].apply(lambda col: pd.factorize(col)[0])

    features = pd.concat(
        [encoded_categories.reset_index(drop=True), pd.DataFrame(title_vectors)], axis=1
    )
    features.columns = features.columns.astype(str)

    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    feature_df = pd.DataFrame(normalized_features, columns=[str(i) for i in range(normalized_features.shape[1])])

    target = merged_data['Average-Rating']

    X_train, X_test, y_train, y_test = train_test_split(feature_df, target, test_size=0.2, random_state=42)

    model = SGDRegressor()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mean_abs_error = mean_absolute_error(y_test, predictions)
    print(f'MAE модели: {mean_abs_error}')

    with open("linreg.pkl", "wb") as file:
        pickle.dump(model, file)

    return merged_data, features

# books = pd.read_csv("Books.csv")
# ratings = pd.read_csv("Ratings.csv")
# modeling(books_preprocessing(books), ratings_preprocessing(ratings))