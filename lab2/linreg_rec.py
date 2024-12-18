import pickle
import re

import joblib
import nltk
import numpy as np
import pandas as pd
import sklearn

from nltk.corpus import stopwords
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

nltk.download("stopwords")
nltk.download("punkt")


def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"], inplace=True)

    def shift_row_right(row, start_index, columns):
        for i in range(len(columns) - 1, start_index, -1):
            row[columns[i]] = row[columns[i - 1]]
        row[columns[start_index]] = np.nan
        return row

    invalid_rows = df[df["Year-Of-Publication"].str.match("[^0-9]", na=False)]

    columns = df.columns.tolist()
    start_index = columns.index("Book-Author")

    df.loc[invalid_rows.index] = df.loc[invalid_rows.index].apply(
        shift_row_right,
        axis=1,
        start_index=start_index,
        columns=columns
    )

    df["Year-Of-Publication"] = pd.to_numeric(df["Year-Of-Publication"], errors="coerce")

    return df


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Book-Rating"] > 0].copy()

    book_counts = df["ISBN"].value_counts()
    valid_books = book_counts[book_counts > 1].index
    df = df[df["ISBN"].isin(valid_books)]

    user_counts = df["User-ID"].value_counts()
    valid_users = user_counts[user_counts > 1].index
    df = df[df["User-ID"].isin(valid_users)]

    return df


def title_preprocessing(text: str) -> str:
    """Функция для нормализации текстовых данных в стобце Book-Title:
    - токенизация
    - удаление стоп-слов
    - удаление пунктуации
    Опционально можно убрать шаги или добавить дополнительные.
    """

    pass


def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    book_mean_rating = ratings.groupby("ISBN")["Book-Rating"].mean().reset_index().rename(
        columns={"Book-Rating": "Mean-Rating"})
    data_for_linreg = books.merge(book_mean_rating, on="ISBN", how="inner")

    data_for_linreg.dropna(subset=["Mean-Rating"], inplace=True)

    X = data_for_linreg[["Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"]]
    y = data_for_linreg["Mean-Rating"]

    title_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    categorical_encoder = OneHotEncoder(handle_unknown='ignore')

    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(transformers=[
        ("title", title_vectorizer, "Book-Title"),
        ("author", categorical_encoder, ["Book-Author"]),
        ("publisher", categorical_encoder, ["Publisher"]),
        ("year", numeric_transformer, ["Year-Of-Publication"])
    ])

    model_linreg = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", SGDRegressor(random_state=42, max_iter=1000, tol=1e-3))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_linreg.fit(X_train, y_train)

    y_pred = model_linreg.predict(X_test)
    score = mean_absolute_error(y_test, y_pred)

    print(score)

    with open("linreg.pkl", "wb") as file:
        pickle.dump(model_linreg, file)
