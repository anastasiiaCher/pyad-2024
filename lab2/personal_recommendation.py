import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from scipy.sparse import csr_matrix

ratings = pd.read_csv("Ratings.csv")
books = pd.read_csv("Books.csv")

filtered_ratings = ratings[ratings['Book-Rating'] > 4]
book_counts = filtered_ratings['ISBN'].value_counts()

# Количество оценок для каждого пользователя
user_counts = filtered_ratings['User-ID'].value_counts()

books_to_keep = book_counts[book_counts > 1].index  # книги с более чем одной оценкой
users_to_keep = user_counts[user_counts > 1].index  # пользователи с более чем одной оценкой

filtered_ratings = ratings[
    (ratings['ISBN'].isin(books_to_keep)) &
    (ratings['User-ID'].isin(users_to_keep))
]

merged_data = pd.merge(books, filtered_ratings, on='ISBN', how='left')

merged_data = merged_data[merged_data['Book-Rating'] != 0]

# Оставляем только нужные столбцы
columns_to_use = ['User-ID', 'ISBN', 'Book-Author', 'Publisher', 'Year-Of-Publication', 'Book-Title', 'Book-Rating']
merged_data = merged_data[columns_to_use]

merged_data = merged_data.dropna()

data = pd.merge(ratings, books, on='ISBN', how='inner')

one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
author_publisher_encoded = one_hot_encoder.fit_transform(merged_data[['Book-Author', 'Publisher']])
svd = joblib.load('svd.pkl')
linreg = joblib.load('sgd_regressor.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
scaler = joblib.load('scaler.pkl')

def generate_personal_recommendations(svd, linreg, tfidf_vectorizer, scaler, data, books):
    user_zero_ratings = data[data['Book-Rating'] == 0]['User-ID'].value_counts()
    user_with_most_zeros = user_zero_ratings.idxmax()

    target_user_ratings = data[data['User-ID'] == user_with_most_zeros]
    unrated_isbns = set(books['ISBN']) - set(target_user_ratings['ISBN'])

    predictions_svd = []
    for isbn in unrated_isbns:
        try:
            prediction = svd.predict(user_with_most_zeros, isbn)
            predictions_svd.append((isbn, prediction.est))
        except:
            continue  # Игнорируем ошибки предсказания

    # Отбор книг с прогнозируемым рейтингом не ниже 8
    recommended_books_svd = [(isbn, est) for isbn, est in predictions_svd if est >= 8]

    if not recommended_books_svd:
        print(f"Для пользователя {user_with_most_zeros} нет книг с SVD рейтингом >= 8.")
        return []

    recommendations_linreg = []
    for isbn, svd_rating in recommended_books_svd:
        book_row = books[books['ISBN'] == isbn]
        if book_row.empty:
            continue

        # Извлекаем признаки книги
        title_tfidf = tfidf_vectorizer.transform(book_row['Book-Title'].values.astype(str))
        year_scaled = scaler.transform(book_row[['Year-Of-Publication']].values.astype(float))
        author_publisher_encoded = one_hot_encoder.transform(book_row[['Book-Author', 'Publisher']])

        # Объединяем признаки
        features = hstack([title_tfidf, author_publisher_encoded, csr_matrix(year_scaled)])

        # Предсказание линейной модели
        predicted_rating = linreg.predict(features)[0]
        recommendations_linreg.append((isbn, predicted_rating))

    sorted_recommendations = sorted(recommendations_linreg, key=lambda x: x[1], reverse=True)

    recommendation_text = f'# Рекомендации для пользователя {user_with_most_zeros}:\n'
    for isbn, rating in sorted_recommendations:
        recommendation_text += f'# ISBN: {isbn}, Предсказанный рейтинг: {rating:.2f}\n'

    print(recommendation_text)

generate_personal_recommendations(svd, linreg, tfidf_vectorizer, scaler, data, books)
