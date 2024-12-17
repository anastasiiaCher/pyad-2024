import pickle
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Books.scv"""
    df = df.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])
    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')
    df = df[df['Year-Of-Publication'] <= 2016]
    df = df.dropna()
    return df


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv
    Целевой переменной в этой задаче будет средний рейтинг книги,
    поэтому в предобработку (помимо прочего) нужно включить:
    1. Замену оценки книги пользователем на среднюю оценку книги всеми пользователями.
    2. Расчет числа оценок для каждой книги (опционально)."""

    df = df[df['Book-Rating'] > 0]
    avg_ratings = df.groupby('ISBN')['Book-Rating'].mean().reset_index()
    avg_ratings.rename(columns={'Book-Rating': 'Average-Rating'}, inplace=True)
    return avg_ratings


def title_preprocessing(text: str) -> str:
    """Функция для нормализации текстовых данных в стобце Book-Title:
    - токенизация
    - удаление стоп-слов
    - удаление пунктуации
    Опционально можно убрать шаги или добавить дополнительные.
    """
    #У меня были проблемы с nltk, которые никак не получалось решить =(
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    data = pd.merge(books, ratings, on='ISBN', how='inner')
    data['Book-Title'] = data['Book-Title'].apply(title_preprocessing)
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    title_vectors = tfidf.fit_transform(data['Book-Title']).toarray()
    title_columns = [f"Title_{i}" for i in range(1000)]
    label_encoder_author = LabelEncoder()
    label_encoder_publisher = LabelEncoder()
    data['Book-Author'] = label_encoder_author.fit_transform(data['Book-Author'])
    data['Publisher'] = label_encoder_publisher.fit_transform(data['Publisher'])
    X = pd.DataFrame(title_vectors, columns=title_columns)
    X['Book-Author'] = data['Book-Author']
    X['Publisher'] = data['Publisher']
    X['Year-Of-Publication'] = data['Year-Of-Publication']

    if X.shape[1] < 1004:
        X['Dummy'] = 0.0
    y = data['Average-Rating']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = SGDRegressor(random_state=42, max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)

    with open("linreg.pkl", "wb") as f:
        pickle.dump(model, f)
