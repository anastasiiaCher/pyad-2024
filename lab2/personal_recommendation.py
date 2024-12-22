import pandas as pd
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # Исправляем строку 209538
    df.loc[209538, "Book-Title"] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
    df.loc[209538, "Book-Author"] = "Michael Teitelbaum"
    df.loc[209538, "Year-Of-Publication"] = "2000"
    df.loc[209538, "Publisher"] = "DK Publishing Inc"

    # Исправляем строку 220731
    df.loc[220731, "Book-Title"] = "Peuple du ciel, suivi de 'Les Bergers'"
    df.loc[220731, "Book-Author"] = "Jean-Marie Gustave Le Clézio"
    df.loc[220731, "Year-Of-Publication"] = "2003"
    df.loc[220731, "Publisher"] = "Gallimard"

    # Исправляем строку 221678
    df.loc[221678, "Book-Title"] = "DK Readers: Creating the X-Men, How Comic Book Artists Work (Level 4: Proficient Readers)"
    df.loc[221678, "Book-Author"] = "Michael Teitelbaum"
    df.loc[221678, "Year-Of-Publication"] = "2000"
    df.loc[221678, "Publisher"] = "DK Publishing Inc"

    # Получим текущий год
    current_year = 2024

    # Фильтрация данных
    df = df[(df["Year-Of-Publication"].astype(str).str.isdigit())]  # Оставляем только числовые значения
    df["Year-Of-Publication"] = df["Year-Of-Publication"].astype(int)  # Приводим к целым числам
    df = df[df["Year-Of-Publication"] <= current_year]  # Удаляем будущие годы

    # Удаление пропусков в критически важных столбцах
    df = df.dropna(subset=["Book-Author", "Publisher"])

    # Удаление столбцов с картинками
    df = df.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"])

    return df


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:

    # Исключаем книги, которые были оценены только один раз
    book_counts = df['ISBN'].value_counts()
    df = df[df['ISBN'].isin(book_counts[book_counts > 1].index)]

    # Исключаем пользователей, которые поставили только одну оценку
    user_counts = df['User-ID'].value_counts()
    df = df[df['User-ID'].isin(user_counts[user_counts > 1].index)]

    # Удаляем строки с пропусками
    df.dropna(inplace=True)

    return df


def title_preprocessing(text: str) -> str:
    stop_words = set(nltk.corpus.stopwords.words("english"))
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    tokens = [word for word in text.lower().split() if word not in stop_words]
    return " ".join(tokens)

books = books_preprocessing(pd.read_csv("Books.csv"))
ratings = ratings_preprocessing(pd.read_csv("Ratings.csv"))

with open("svd.pkl", "rb") as f:
    svd = pickle.load(f)
with open("linreg.pkl", "rb") as f:
    sgd = pickle.load(f)

book_avg_ratings = ratings.groupby("ISBN")["Book-Rating"].mean()

merged_data = books[books["ISBN"].isin(book_avg_ratings.index)]
merged_data["Average-Rating"] = merged_data["ISBN"].map(book_avg_ratings)

merged_data = merged_data.dropna(
    subset=["Average-Rating", "Book-Author", "Publisher", "Year-Of-Publication", "Book-Title"])

merged_data['Book-Title'] = merged_data['Book-Title'].apply(title_preprocessing)
vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
vectorized_titles = vectorizer.fit_transform(merged_data['Book-Title']).toarray()

categorical_features = merged_data[['Book-Author', 'Publisher', 'Year-Of-Publication']]
categorical_encoded = pd.DataFrame({
    col: pd.factorize(categorical_features[col])[0]
    for col in categorical_features
})

features = pd.concat([categorical_encoded, pd.DataFrame(vectorized_titles)], axis=1)
features.columns = features.columns.astype(str)
scaler = StandardScaler()
X = scaler.fit_transform(features)
features = pd.DataFrame(X, columns=features.columns, index=features.index)

most_zeros_user = ratings[ratings['Book-Rating'] == 0]['User-ID'].value_counts().idxmax()
zero_rated_books = ratings[
    (ratings['User-ID'] == most_zeros_user) & (ratings['Book-Rating'] == 0)]['ISBN']

result = []
merged_data = merged_data.reset_index(drop=True)
features = features.reset_index(drop=True)
for item_id in zero_rated_books:
    svd_pred = svd.predict(most_zeros_user, item_id).est
    if svd_pred >= 8:
        indices = merged_data.index[merged_data['ISBN'] == str(item_id)]
        if len(indices) > 0:
            book_features = features.loc[indices].to_numpy()
            linreg_pred = sgd.predict(book_features)[0]
            result.append((item_id, svd_pred, linreg_pred))

result.sort(key=lambda x: x[2], reverse=True)

with open("personal_recommendation.txt", "w+") as rec_file:
    for item_id, svd_pred, linreg_pred in result:
        book_title = books.loc[books['ISBN'] == item_id, 'Book-Title'].values[0]
        rec_file.write(f"Name: {book_title}, SVD rating: {svd_pred:.2f}, LinReg rating: {linreg_pred:.2f}\n\n")