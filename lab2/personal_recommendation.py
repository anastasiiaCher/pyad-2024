import pandas as pd
import joblib

ratings = pd.read_csv("Ratings.csv")
svd = joblib.load("svd.pkl")
model_linreg = joblib.load("linreg.pkl")
books = pd.read_csv("Books.csv")

zero_ratings_count = ratings[ratings["Book-Rating"] == 0].groupby("User-ID")["Book-Rating"].count()
most_zero_user = zero_ratings_count.idxmax()

zero_books = ratings[(ratings["User-ID"] == most_zero_user) & (ratings["Book-Rating"] == 0)]["ISBN"].unique()
preds = svd.test([(most_zero_user, isbn, 0) for isbn in zero_books])
recommended_svd = [pred.iid for pred in preds if pred.est >= 8]

recommended_linreg = books[books["ISBN"].isin(recommended_svd)].copy()
recommended_linreg["Predicted-Mean-Rating"] = model_linreg.predict(recommended_linreg[["Book-Title","Book-Author","Publisher","Year-Of-Publication"]])
recommended_linreg.sort_values("Predicted-Mean-Rating", ascending=False, inplace=True)

recommended_books = recommended_linreg[["ISBN","Book-Title","Book-Author","Predicted-Mean-Rating"]].head(10)

print(recommended_books)
