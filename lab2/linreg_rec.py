import pickle
import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

nltk.download("stopwords")
nltk.download("punkt")


def fix_merged_columns(row):
    merged_value = row['Book-Title']
    pattern = r'^(.+?)\\";(.+)"$'
    match = re.match(pattern, merged_value)
    if match:
        author, title = match.groups()
        return author.strip(), title.strip()
    else:
        return None, None


def books_preprocessing(books: pd.DataFrame) -> pd.DataFrame:
    new_book = books[books["Year-Of-Publication"].map(str).str.match("[^0-9]")]
    books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(str)
    mask = ~books["Year-Of-Publication"].str.contains(r'\D')
    books_filtered = books[mask]
    new_book['Book-Title'].unique()
    fixed_data = new_book.apply(lambda x: fix_merged_columns(x), axis=1)
    new_book['Publisher'] = new_book['Year-Of-Publication']
    new_book['Year-Of-Publication'] = new_book['Book-Author']
    new_book[['Book-Title', 'Book-Author']] = fixed_data.tolist()
    books = pd.concat([books_filtered, new_book])
    books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], downcast='integer')
    books = books.drop('Image-URL-S', axis=1)
    books = books.drop('Image-URL-M', axis=1)
    books = books.drop('Image-URL-L', axis=1)
    return books
    pass


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    ratings = df.rename(columns={'Book-Rating': 'Rating'})
    ratings['Rating'] = ratings['Rating'].astype(float)
    ratings = ratings.query("Rating != 0.0")
    min_ratings = 2
    book_counts = ratings.groupby('ISBN')['User-ID'].nunique()
    user_counts = ratings.groupby('User-ID')['ISBN'].nunique()
    good_books = book_counts[book_counts >= min_ratings].index
    good_users = user_counts[user_counts >= min_ratings].index
    filtered1_ratings = ratings[(ratings['ISBN'].isin(good_books)) & (ratings['User-ID'].isin(good_users))]
    return filtered1_ratings
    pass


def title_preprocessing(text: str) -> str:
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords and token.isalpha()]
    return ' '.join(filtered_tokens)
    pass


def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    categorical_features = ['Book-Author', 'Publisher']
    numerical_features = ['Year-Of-Publication']
    for feature in categorical_features:
        books[feature] = books[feature].astype('category').cat.codes
    avg_ratings = ratings[['ISBN', 'Rating']].drop_duplicates()
    final_df = books.merge(avg_ratings, on='ISBN', how='inner')
    X = final_df[numerical_features + categorical_features]
    y = final_df['Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=10)
    vectorizer = TfidfVectorizer(max_features=5)
    titles = final_df['Book-Title']  # .apply(title_preprocessing)
    X_titles_train = vectorizer.fit_transform(titles.iloc[X_train.index])
    X_titles_test = vectorizer.transform(titles.iloc[X_test.index])
    X_train = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(X_titles_train.toarray())], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(X_titles_test.toarray())], axis=1)
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    #  X_test_scaled = scaler.transform(X_test)
    linreg = LogisticRegression()
    linreg.fit(X_train_scaled, y_train)
    with open("linreg.pkl", "wb") as file:
        pickle.dump(linreg, file)


books1 = pd.read_csv("Books.csv")
ratings1 = pd.read_csv("Ratings.csv")
filtered_ratings1 = ratings_preprocessing(ratings1)
filtered_books1 = books_preprocessing(books1)
modeling(filtered_books1, filtered_ratings1)
