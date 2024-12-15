import pandas as pd
import joblib
import json
from surprise import Reader, Dataset
from tabulate import tabulate

ratings_full = pd.read_csv("Ratings.csv")
svd = joblib.load("svd.pkl")
model_linreg = joblib.load("linreg.pkl")
books = pd.read_csv("Books.csv")

zero_ratings_count = ratings_full[ratings_full["Book-Rating"] == 0].groupby("User-ID")["Book-Rating"].count()
target_user = zero_ratings_count.idxmax()

zero_books = ratings_full[(ratings_full["User-ID"] == target_user) & (ratings_full["Book-Rating"] == 0)]["ISBN"].unique()

testset = [(target_user, isbn, 0) for isbn in zero_books]
predictions = svd.test(testset)

candidate_books = [pred.iid for pred in predictions if pred.est >= 8]

candidate_data = books[books["ISBN"].isin(candidate_books)].copy()
X_candidates = candidate_data[["Book-Title","Book-Author","Publisher","Year-Of-Publication"]]
pred_linreg = model_linreg.predict(X_candidates)
candidate_data["Predicted-Mean-Rating"] = pred_linreg
candidate_data.sort_values("Predicted-Mean-Rating", ascending=False, inplace=True)

recommended_books = candidate_data[["ISBN","Book-Title","Book-Author","Predicted-Mean-Rating"]].head(10)

svd_predictions_df = pd.DataFrame(
    [(pred.uid, pred.iid, pred.est) for pred in predictions if pred.iid in candidate_books],
    columns=["User-ID", "ISBN", "SVD_Predicted_Rating"]
)

print("Таблица предсказаний SVD:")
print(svd_predictions_df.to_markdown(index=False))

print("\nТаблица предсказаний линейной регрессии:")
print(recommended_books.to_markdown(index=False))