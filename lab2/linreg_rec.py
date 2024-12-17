import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")


def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Удаление ненужных столбцов
    df.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], inplace=True)
    # Преобразование некорректных годов
    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')
    df['Year-Of-Publication'] = df['Year-Of-Publication'].fillna(df['Year-Of-Publication'].median())
    return df


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Исключаем нулевые рейтинги
    df = df[df['Book-Rating'] != 0]
    # Заменяем оценки на средние для каждой книги
    avg_ratings = df.groupby('ISBN')['Book-Rating'].mean()
    df['Book-Rating'] = df['ISBN'].map(avg_ratings)
    return df


def title_preprocessing(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    text = re.sub(r'[^\w\s]', '', text)  
    tokens = nltk.word_tokenize(text.lower())  
    tokens = [word for word in tokens if word not in stop_words]  
    return " ".join(tokens)


def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    le_author = LabelEncoder()
    books['Book-Author'] = le_author.fit_transform(books['Book-Author'].fillna('Unknown'))

    le_publisher = LabelEncoder()
    books['Publisher'] = le_publisher.fit_transform(books['Publisher'].fillna('Unknown'))

    # Применение TF-IDF к названиям книг
    tfidf = TfidfVectorizer(max_features=1000)
    books['Book-Title'] = books['Book-Title'].apply(title_preprocessing)
    tfidf_matrix = tfidf.fit_transform(books['Book-Title']).toarray()

    # Объединение данных
    merged = ratings.merge(books, on='ISBN')
    features = ['Book-Author', 'Publisher', 'Year-Of-Publication']
    X = pd.concat([merged[features], pd.DataFrame(tfidf_matrix)], axis=1)
    y = merged['Book-Rating']

    # Загрузка тестового набора для выбора правильных признаков
    test_data = pd.read_csv("linreg_test.csv")
    test_features = test_data.columns.drop("y")  

    X = X.reindex(columns=test_features, fill_value=0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linreg = SGDRegressor(random_state=42)
    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae}")

    with open("linreg.pkl", "wb") as file:
        pickle.dump(linreg, file)

if __name__ == "__main__":
    books = pd.read_csv("Books.csv", low_memory=False)
    ratings = pd.read_csv("Ratings.csv")

    books = books_preprocessing(books)
    ratings = ratings_preprocessing(ratings)

    modeling(books, ratings)