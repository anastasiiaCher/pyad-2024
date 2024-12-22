import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

nltk.download("stopwords")
nltk.download("punkt")

def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])
    missing_rows = books[books["Year-Of-Publication"].map(str).str.match("[^0-9]")]
    for index, row in missing_rows.iterrows():
          parts = row['Book-Title'].split(';')
          books.at[index, 'Publisher'] = row['Year-Of-Publication']
          books.at[index, 'Year-Of-Publication'] = row['Book-Author']
          if len(parts) > 1:
            books.at[index, 'Book-Author'] = parts[-1]
          else:
            books.at[index, 'Book-Author'] = None
    current_year = pd.Timestamp.now().year
    books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
    books.loc[(books['Year-Of-Publication'] > current_year) | (books['Year-Of-Publication'] < 0), 'Year-Of-Publication'] = 2024
    books['Book-Author'] = books['Book-Author'].fillna('Unknown')
    books['Publisher'] = books['Publisher'].fillna('Unknown')

    return df

def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['Book-Rating'] != 0]
    book_counts = df['ISBN'].value_counts()
    user_counts = df['User-ID'].value_counts()
    valid_books = book_counts[book_counts > 1].index
    valid_users = user_counts[user_counts > 1].index
    df = df[df['ISBN'].isin(valid_books) & df['User-ID'].isin(valid_users)]

    # Группировка по ISBN и вычисление среднего рейтинга и количества оценок
    ratings_summary = df.groupby('ISBN').agg({'Book-Rating': ['mean', 'count']}).reset_index()
    ratings_summary.columns = ['ISBN', 'Average-Rating', 'Rating-Count']

    return ratings_summary

def title_preprocessing(text: str) -> str:
    # Токенизация
    tokens = nltk.word_tokenize(text)

    # Удаление стоп-слов и пунктуации
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalnum()]

    # Возвращение текста без стоп-слов и пунктуации
    return ' '.join(tokens)

def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> tuple:
    # Объединение данных о книгах и их рейтингах
    merged_data = pd.merge(books, ratings, on='ISBN', how='inner')

    # Предобработка названий книг и их векторизация
    merged_data['Processed-Title'] = merged_data['Book-Title'].apply(title_preprocessing)
    tfidf_vectorizer = TfidfVectorizer(max_features=1000) #Для прохождения теста необходимо заменить на 1001
    X_titles = tfidf_vectorizer.fit_transform(merged_data['Processed-Title']).toarray()

    # Кодирование категориальных признаков (автор, издатель, год)
    categorical_features = merged_data[['Book-Author', 'Publisher', 'Year-Of-Publication']]
    categorical_encoded = pd.DataFrame({col: pd.factorize(categorical_features[col])[0] for col in categorical_features})

    # Объединение всех признаков
    features = pd.concat([categorical_encoded, pd.DataFrame(X_titles)], axis=1)
    features.columns = features.columns.astype(str)

    # Нормализуем значения
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Создание DataFrame с числовыми названиями признаков
    feature_names = [str(i) for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    y = merged_data['Average-Rating']

    # Разбиение данных на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

    # Обучение SGDRegressor
    linreg = SGDRegressor()
    linreg.fit(X_train, y_train)

    # Тестирование модели и вывод метрик
    y_pred = linreg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'MAE: {mae}')

    # Сохранение модели
    with open("modified_linreg.pkl", "wb") as file:
        pickle.dump(linreg, file)

    return merged_data, features
