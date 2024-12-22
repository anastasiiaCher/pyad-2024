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


def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Books.scv"""
    df = df.copy()

    new_sort = df[df["Year-Of-Publication"].map(str).str.match("[^0-9]")]
    df['Year-Of-Publication'] = df['Year-Of-Publication'].astype(str)
    mask = ~df["Year-Of-Publication"].str.contains(r'\D')
    books_filtered = df[mask]
    
    # Применяем функцию исправления данных
    fixed_df = new_sort.apply(lambda x: fix_merged_columns(x), axis=1)
    new_sort['Publisher'] = new_sort['Year-Of-Publication']
    new_sort['Year-Of-Publication'] = new_sort['Book-Author']
    new_sort[['Book-Title', 'Book-Author']] = fixed_df.tolist()
    
    df = pd.concat([books_filtered, new_sort])

    # Преобразование и очистка некорректных дат в столбце "Year-Of-Publication"
    df['Year-Of-Publication'] = pd.to_datetime(df['Year-Of-Publication'], errors='coerce')
    df = df.dropna(subset=['Year-Of-Publication'])
    
    # Удаляем строки с будущими годами публикации
    current_year = pd.to_datetime('today').year
    df = df[df['Year-Of-Publication'].dt.year <= current_year]  # Сравниваем только год
    
    # Удаление ненужных столбцов с изображениями
    df = df.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])

    df['Book-Author'] = df['Book-Author'].fillna('Unknown')
    df['Publisher'] = df['Publisher'].fillna('Unknown')

    return df


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv
    Целевой переменной в этой задаче будет средний рейтинг книги,
    поэтому в предобработку (помимо прочего) нужно включить:
    1. Замену оценки книги пользователем на среднюю оценку книги всеми пользователями.
    2. Расчет числа оценок для каждой книги (опционально)."""
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


def title_preprocessing(text: str) -> str:
    """Функция для нормализации текстовых данных в стобце Book-Title:
    - токенизация
    - удаление стоп-слов
    - удаление пунктуации
    Опционально можно убрать шаги или добавить дополнительные.
    """
    stop_words = set(stopwords.words('english'))
    # Токенизация (разделение на слова)
    tokens = nltk.word_tokenize(text.lower())
    # Удаление стоп-слов и пунктуации
    tokens = [word for word in tokens if word not in stop_words]

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
    # Преобразование авторов и издателей с использованием LabelEncoder
    author_encoder = LabelEncoder()
    books['Book-Author'] = author_encoder.fit_transform(books['Book-Author'].fillna('Unknown'))

    publisher_encoder = LabelEncoder()
    books['Publisher'] = publisher_encoder.fit_transform(books['Publisher'].fillna('Unknown'))

    # Применение TF-IDF для названий книг
    vectorizer = TfidfVectorizer(max_features=1000)
    books['Book-Title'] = books['Book-Title'].apply(title_preprocessing)
    tfidf_matrix = vectorizer.fit_transform(books['Book-Title']).toarray()

    # Объединение данных по ISBN
    merged_data = ratings.merge(books, on='ISBN')
    selected_features = ['Book-Author', 'Publisher', 'Year-Of-Publication']
    X = pd.concat([merged_data[selected_features], pd.DataFrame(tfidf_matrix)], axis=1)
    y = merged_data['Book-Rating']

    # Загрузка тестового набора для синхронизации признаков
    test_data = pd.read_csv("linreg_test.csv")
    test_features = test_data.columns.drop("y")

    X = X.reindex(columns=test_features, fill_value=0)

    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Сохранение масштабатора
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Обучение модели линейной регрессии с использованием стохастического градиентного спуска
    model = SGDRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Прогнозирование и вычисление MAE
    y_pred = model.predict(X_test)
    mae_score = mean_absolute_error(y_test, y_pred)
    print(f"Средняя абсолютная ошибка (MAE): {mae_score}")

    # Сохранение обученной модели
    with open("linreg.pkl", "wb") as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    # Загрузка данных
    books_data = pd.read_csv("Books.csv", low_memory=False)
    ratings_data = pd.read_csv("Ratings.csv")

    # Предобработка данных
    books_data = books_preprocessing(books_data)
    ratings_data = ratings_preprocessing(ratings_data)

    # Запуск модели
    model_books_ratings(books_data, ratings_data)
