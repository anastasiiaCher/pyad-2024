import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDRegressor
from scipy.sparse import hstack
from surprise import SVD
from surprise import Dataset
from surprise import Reader
import pickle
import sys


ratings = pd.read_csv("Ratings.csv")
books = pd.read_csv("Books.csv")

books["Year-Of-Publication"] = books["Year-Of-Publication"].astype(str)

shifted_rows = books[books["Year-Of-Publication"].str.match("[^0-9]", na=False)]

column_names = books.columns.tolist()

shift_start_index = column_names.index("Book-Author")

for row_index in shifted_rows.index:
    row_data = books.loc[row_index]

    for col_index in range(len(column_names) - 1, shift_start_index, -1):
        row_data[column_names[col_index]] = row_data[column_names[col_index - 1]]

    row_data[column_names[shift_start_index]] = np.nan

    books.loc[row_index] = row_data
books = books[books["Year-Of-Publication"].astype(int) <= 2024]
books = books.iloc[:, :-3]

user_with_most_zeros = ratings.groupby("User-ID").filter(lambda x: len(x[x["Book-Rating"] == 0]) > 0).groupby("User-ID").size().idxmax()

user_ratings = ratings[ratings["User-ID"] == user_with_most_zeros]
unrated_books = books[~books["ISBN"].isin(user_ratings["ISBN"])]

unrated_books_svd = unrated_books.copy()
unrated_books_svd["Estimated-Rating"] = unrated_books_svd["ISBN"].apply(
    lambda isbn: svd.predict(user_with_most_zeros, isbn).est
)

top_books_svd = unrated_books_svd[unrated_books_svd["Estimated-Rating"] >= 8]

top_books_linreg = top_books_svd.copy()

X_title = vectorizer.transform(top_books_svd["Book-Title"])
X_author = label_encoder_author.transform(top_books_svd["Book-Author"]).reshape(-1, 1)
X_publisher = label_encoder_publisher.transform(top_books_svd["Publisher"]).reshape(-1, 1)
X_year = scaler_year.transform(top_books_svd[["Year-Of-Publication"]])

X_combined_linreg = hstack([X_title, X_author, X_publisher, X_year])

top_books_linreg["Predicted-Rating"] = linreg.predict(X_combined_linreg)


recommended_books = top_books_linreg.sort_values(by="Predicted-Rating", ascending=False)

with open("user_recommendations.txt", "w") as rec_file:
    for index, row in recommended_books.iterrows():
        rec_file.write(f"{row['Book-Title']} - Predicted Rating: {row['Predicted-Rating']:.2f}\n")
