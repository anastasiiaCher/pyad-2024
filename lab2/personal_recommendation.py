import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Books.csv"""
    df = df.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])
    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')
    df = df[df['Year-Of-Publication'] <= 2016]
    df = df.dropna()
    return df


ratings_full = pd.read_csv("Ratings.csv")
svd = joblib.load("svd.pkl")
model_linreg = joblib.load("linreg.pkl")
books_df = pd.read_csv("Books.csv", low_memory=False)
books = books_preprocessing(books_df)
zero_ratings_count = ratings_full[ratings_full["Book-Rating"] == 0].groupby("User-ID")["Book-Rating"].count()
target_user = zero_ratings_count.idxmax()
zero_books = ratings_full[
    (ratings_full["User-ID"] == target_user) & (ratings_full["Book-Rating"] == 0)
    ]["ISBN"].unique()

testset = [(target_user, isbn, 0) for isbn in zero_books]
predictions = svd.test(testset)
candidate_books = [pred.iid for pred in predictions if pred.est >= 8]
candidate_data = books[books["ISBN"].isin(candidate_books)].copy()
candidate_data.loc[:, 'Book-Author'] = candidate_data['Book-Author'].fillna('Unknown')
candidate_data.loc[:, 'Publisher'] = candidate_data['Publisher'].fillna('Unknown')
candidate_data.loc[:, 'Year-Of-Publication'] = candidate_data['Year-Of-Publication'].fillna(0)
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
title_vectors = tfidf.fit_transform(books["Book-Title"]).toarray()
candidate_titles = tfidf.transform(candidate_data["Book-Title"]).toarray()
candidate_features = pd.DataFrame(candidate_titles, columns=[f"Title_{i}" for i in range(candidate_titles.shape[1])])
candidate_features["Book-Author"] = candidate_data["Book-Author"].factorize()[0]
candidate_features["Publisher"] = candidate_data["Publisher"].factorize()[0]
candidate_features["Year-Of-Publication"] = candidate_data["Year-Of-Publication"]
candidate_features = candidate_features.fillna(0)
while candidate_features.shape[1] < 1004:
    candidate_features[f"Dummy_{candidate_features.shape[1]}"] = 0.0
scaler = StandardScaler()
candidate_features_scaled = scaler.fit_transform(candidate_features)
pred_linreg = model_linreg.predict(candidate_features_scaled)
candidate_data["Predicted-Mean-Rating"] = pred_linreg
candidate_data.sort_values("Predicted-Mean-Rating", ascending=False, inplace=True)
recommended_books = candidate_data[["ISBN", "Book-Title", "Book-Author", "Predicted-Mean-Rating"]].head(10)
svd_predictions_df = pd.DataFrame(
    [(pred.uid, pred.iid, pred.est) for pred in predictions if pred.iid in candidate_books],
    columns=["User-ID", "ISBN", "SVD_Predicted_Rating"]
)
'''
User-ID: 198711

Таблица рекомендаций:
| ISBN       | Book-Title                                                            | Book-Author         |   Predicted-Mean-Rating |
|:-----------|:----------------------------------------------------------------------|:--------------------|------------------------:|
| 0440406498 | The Black Cauldron (Chronicles of Prydain (Paperback))                | LLOYD ALEXANDER     |                 8.06628 |
| 0440940001 | Island of the Blue Dolphins (Laurel Leaf Books)                       | Scott O'Dell        |                 7.94352 |
| 0440901588 | A Swiftly Tilting Planet                                              | Madeleine L'Engle   |                 7.8321  |
| 059035342X | Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))      | J. K. Rowling       |                 7.71788 |
| 0439064872 | Harry Potter and the Chamber of Secrets (Book 2)                      | J. K. Rowling       |                 7.70829 |
'''
