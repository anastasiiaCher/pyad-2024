import pandas as pd
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

nltk.download("stopwords")
nltk.download("punkt")

def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Books.scv"""
    df = df.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])

    missing_rows = df[df["Year-Of-Publication"].map(str).str.match("[^0-9]")]
    for index, row in missing_rows.iterrows():
        parts = row['Book-Title'].split(';')
        df.at[index, 'Publisher'] = row['Year-Of-Publication']
        df.at[index, 'Year-Of-Publication'] = row['Book-Author']
        if len(parts) > 1:
            df.at[index, 'Book-Author'] = parts[-1]
        else:
            df.at[index, 'Book-Author'] = None

    current_year = pd.Timestamp.now().year
    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')
    df.loc[(df['Year-Of-Publication'] > current_year) | (df['Year-Of-Publication'] < 0), 'Year-Of-Publication'] = 2024

    df['Book-Author'] = df['Book-Author'].fillna('Unknown')
    df['Publisher'] = df['Publisher'].fillna('Unknown')
    return df


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv
    Целевой переменной в этой задаче будет средний рейтинг книги,
    поэтому в предобработку (помимо прочего) нужно включить:
    1. Замену оценки книги пользователем на среднюю оценку книги всеми пользователями.
    2. Расчет числа оценок для каждой книги (опционально)."""
    df = df[df['Book-Rating'] > 0]

    book_counts = df["ISBN"].value_counts()
    df = df[df["ISBN"].isin(book_counts[book_counts > 1].index)]

    user_counts = df["User-ID"].value_counts()
    df = df[df["User-ID"].isin(user_counts[user_counts > 1].index)]

    return df


def title_preprocessing(text: str) -> str:
    """Функция для нормализации текстовых данных в стобце Book-Title:
    - токенизация
    - удаление стоп-слов
    - удаление пунктуации
    Опционально можно убрать шаги или добавить дополнительные.
    """
    stop_words = set(nltk.corpus.stopwords.words("english"))
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    tokens = [word for word in text.lower().split() if word not in stop_words]
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
    books = books_preprocessing(books)
    ratings = ratings_preprocessing(ratings)

    merged_data = ratings.merge(books, on='ISBN')

    merged_data['Clean-Title'] = merged_data['Book-Title'].apply(title_preprocessing)
    vectorizer = TfidfVectorizer(max_features=100)
    vectorized_titles = vectorizer.fit_transform(merged_data['Clean-Title']).toarray()

    categorical_features = merged_data[['Book-Author', 'Publisher', 'Year-Of-Publication']]
    categorical_encoded = pd.DataFrame({
        col: pd.factorize(categorical_features[col])[0]
        for col in categorical_features
    })

    features = pd.concat([categorical_encoded, pd.DataFrame(vectorized_titles)], axis=1)
    features.columns = features.columns.astype(str)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = merged_data['Book-Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=108)

    test_data = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
    test_data['y'] = y_test.values
    test_data.to_csv("linreg_test.csv", index=False)

    linreg = SGDRegressor()
    linreg.fit(X_train, y_train)

    # Tests
    predictions = linreg.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print("MAE:", mae)

    with open("linreg.pkl", "wb") as file:
        pickle.dump(linreg, file)


if __name__ == "__main__":
    books = pd.read_csv("Books.csv")
    ratings = pd.read_csv("Ratings.csv")
    modeling(books, ratings)
