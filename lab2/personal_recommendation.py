import pandas as pd
import numpy as np
import pickle

from surprise import Dataset, Reader

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

with open("svd_new.pkl", "rb") as f:
    model_svd = pickle.load(f)

with open("linreg_new.pkl", "rb") as f:
    linreg = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("authors_dict.pkl", "rb") as f:
    authors_dict = pickle.load(f)

with open("publishers_dict.pkl", "rb") as f:
    publishers_dict = pickle.load(f)

ratings = pd.read_csv("Ratings.csv", low_memory=False)
books = pd.read_csv("Books.csv", low_memory=False)

books["Year-Of-Publication"] = pd.to_numeric(books["Year-Of-Publication"], errors='coerce')
books = books[(books["Year-Of-Publication"] > 0) & (books["Year-Of-Publication"] <= 2024)].copy()

books = books.dropna(subset=["Book-Author", "Publisher", "Book-Title"]).copy()

cols_to_drop = ["Image-URL-S", "Image-URL-M", "Image-URL-L"]
books.drop(columns=[c for c in cols_to_drop if c in books.columns],
           errors="ignore", inplace=True)

zero_ratings = ratings[ratings["Book-Rating"] == 0]
if zero_ratings.empty:
    print("Нет пользователей с нулевыми рейтингами. Рекомендации не сгенерированы.")
    exit(0)

user_zero_counts = zero_ratings["User-ID"].value_counts()
top_user = user_zero_counts.idxmax()

top_user_zeros = zero_ratings[zero_ratings["User-ID"] == top_user]["ISBN"].unique()

svd_predictions = []
for isbn in top_user_zeros:
    pred = model_svd.predict(uid=top_user, iid=isbn)
    svd_predictions.append((isbn, pred.est))

svd_predictions = pd.DataFrame(svd_predictions, columns=["ISBN", "svd_pred"])

svd_good = svd_predictions[svd_predictions["svd_pred"] >= 8]

if svd_good.empty:
    print(f"Рекомендация: Нет книг с предсказанным SVD-рейтингом >= 8 для пользователя {top_user}.")
    exit(0)

train_ratings = ratings[ratings["Book-Rating"] > 0].copy()

book_counts = train_ratings["ISBN"].value_counts()
valid_books = book_counts[book_counts > 1].index

user_counts = train_ratings["User-ID"].value_counts()
valid_users = user_counts[user_counts > 1].index

train_ratings = train_ratings[train_ratings["ISBN"].isin(valid_books)]
train_ratings = train_ratings[train_ratings["User-ID"].isin(valid_users)]

books_filtered = books[books["ISBN"].isin(train_ratings["ISBN"].unique())].copy()

book_mean_ratings = train_ratings.groupby("ISBN")["Book-Rating"].mean().reset_index(name="mean_rating")
books_merge = pd.merge(books_filtered, book_mean_ratings, on="ISBN", how="inner")

recommend_books = books_merge[books_merge["ISBN"].isin(svd_good["ISBN"])].copy()

if recommend_books.empty:
    print("Рекомендация: Не найдены сведения о книгах, которые SVD отметил рейтингом >=8.")
    exit(0)


def prepare_features(df):
    titles_vec = vectorizer.transform(df["Book-Title"].fillna(""))
    df["author_id"] = df["Book-Author"].map(authors_dict).fillna(-1).astype(int)
    df["publisher_id"] = df["Publisher"].map(publishers_dict).fillna(-1).astype(int)
    year_arr = df["Year-Of-Publication"].fillna(0).values.reshape(-1, 1)

    numeric_data = np.column_stack([
        year_arr,
        df["author_id"],
        df["publisher_id"]
    ])

    numeric_scaled = scaler.transform(numeric_data)

    numeric_sparse = csr_matrix(numeric_scaled)
    X_final = hstack([numeric_sparse, titles_vec], format='csr')
    return X_final


X_rec = prepare_features(recommend_books)
lin_pred = linreg.predict(X_rec)
recommend_books["lin_pred"] = lin_pred

recommend_books = pd.merge(
    recommend_books,
    svd_good,
    on="ISBN",
    how="inner"
)

recommendation = recommend_books.sort_values("lin_pred", ascending=False)

print(f"Рекомендация для пользователя (User-ID={top_user}):")
print("=" * 80)

top_recommendations = recommendation[
    ["ISBN", "Book-Title", "Book-Author", "Publisher", "lin_pred", "svd_pred"]
].head(5)

for idx, row in top_recommendations.iterrows():
    print(f"{idx + 1}. Название книги: {row['Book-Title']}")
    print(f"   Автор: {row['Book-Author']}")
    print(f"   Издатель: {row['Publisher']}")
    print(f"   ISBN: {row['ISBN']}")
    print(f"   Предсказанный рейтинг (LinReg): {row['lin_pred']:.2f}")
    print(f"   Предсказанный рейтинг (SVD): {row['svd_pred']:.2f}")
    print("-" * 80)

"""
Рекомендация для пользователя (User-ID=198711):
================================================================================
7. Название книги: The Hobbit : The Enchanting Prelude to The Lord of the Rings
   Автор: J.R.R. TOLKIEN
   Издатель: Del Rey
   ISBN: 0345339681
   Предсказанный рейтинг (LinReg): 8.53
   Предсказанный рейтинг (SVD): 8.07
--------------------------------------------------------------------------------
26. Название книги: The Lion, the Witch and the Wardrobe (Full-Color Collector's Edition)
   Автор: C. S. Lewis
   Издатель: HarperTrophy
   ISBN: 0064409422
   Предсказанный рейтинг (LinReg): 8.36
   Предсказанный рейтинг (SVD): 8.41
--------------------------------------------------------------------------------
5. Название книги: Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))
   Автор: J. K. Rowling
   Издатель: Arthur A. Levine Books
   ISBN: 059035342X
   Предсказанный рейтинг (LinReg): 8.32
   Предсказанный рейтинг (SVD): 8.10
--------------------------------------------------------------------------------
10. Название книги: Cat in the Hat (I Can Read It All by Myself Beginner Books)
   Автор: Seuss
   Издатель: Random House Books for Young Readers
   ISBN: 0394900014
   Предсказанный рейтинг (LinReg): 8.09
   Предсказанный рейтинг (SVD): 8.17
--------------------------------------------------------------------------------
15. Название книги: Little House in the Big Woods
   Автор: Laura Ingalls Wilder
   Издатель: HarperTrophy
   ISBN: 0064400018
   Предсказанный рейтинг (LinReg): 8.03
   Предсказанный рейтинг (SVD): 8.37
--------------------------------------------------------------------------------
"""
