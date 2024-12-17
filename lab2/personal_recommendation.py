import pandas as pd
import pickle
from linreg_rec import books_preprocessing, ratings_preprocessing, title_preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

books = books_preprocessing(pd.read_csv("Books.csv"))
ratings_original = pd.read_csv("Ratings.csv")
ratings = ratings_preprocessing(ratings_original.copy())

with open("svd.pkl", "rb") as f:
    svd = pickle.load(f)
with open("linreg.pkl", "rb") as f:
    sgd = pickle.load(f)

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

most_zeros_user = ratings_original[ratings_original['Book-Rating'] == 0]['User-ID'].value_counts().idxmax()
zero_rated_books = ratings_original[
    (ratings_original['User-ID'] == most_zeros_user) & (ratings_original['Book-Rating'] == 0)]

recommendations = []
for item_id in zero_rated_books['ISBN']:
    svd_pred = svd.predict(most_zeros_user, item_id).est
    if svd_pred >= 8:
        book_features = features[merged_data['ISBN'] == item_id].to_numpy()
        linreg_pred = sgd.predict(book_features)[0]
        recommendations.append((item_id, svd_pred, linreg_pred))

recommendations.sort(key=lambda x: x[2], reverse=True)

with open("user_recommendations.txt", "w+") as rec_file:
    for item_id, svd_pred, linreg_pred in recommendations:
        book_title = books.loc[books['ISBN'] == item_id, 'Book-Title'].values[0]
        rec_file.write(f"Book: {book_title}\nPredicted rating: {svd_pred:.2f}\nSGD rating: {linreg_pred:.2f}\n\n")
