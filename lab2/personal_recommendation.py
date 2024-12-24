import pandas as pd
import pickle
from linreg_rec import prepare_data, ratings_preprocessing, books_preprocessing

# загружаем исходные данные
ratings = pd.read_csv("Ratings.csv")
books = pd.read_csv("Books.csv")

# загружаем модели
with open('svd.pkl', 'rb') as f:
    loaded_svd = pickle.load(f)

with open('linreg.pkl', 'rb') as f:
    loaded_linreg = pickle.load(f)

# находим юзера с самым большим количество поставленных 0
zero_ratings_count = ratings[ratings["Book-Rating"] == 0].groupby("User-ID")["Book-Rating"].count()
user_id = zero_ratings_count.idxmax()

# находим книги, которым пользователь поставил 0
user_books = ratings[(ratings["User-ID"] == user_id) & (ratings["Book-Rating"] == 0)]

# делаем предсказание SVD для книг, которым пользователь "поставил" 0.
svd_preds = loaded_svd.test([(user_id, isbn, 0) for isbn in user_books["ISBN"]])

# берем те книги, для которых предсказали рейтинг не ниже 8
svd_rec = pd.DataFrame(columns=['ISBN', 'SVD_Rating'])
for pred in svd_preds:
    if pred.est >= 8:
        svd_rec.loc[len(svd_rec)] = [pred.iid, pred.est]

recommended_svd_isbn = svd_rec['ISBN'].values

# делаем предсказание LinReg для этих же книг.
prepared_books = books_preprocessing(books)
prepared_raitings = ratings_preprocessing(ratings)

books_linreg = books[books["ISBN"].isin(recommended_svd_isbn)].copy()
ratings_linreg = ratings[ratings["ISBN"].isin(recommended_svd_isbn)].copy()

X, y, df_isbn  = prepare_data(prepared_books, prepared_raitings)

X = pd.concat([X, df_isbn], axis=1)
X = X[X["ISBN"].isin(recommended_svd_isbn)]
X = X.drop(columns=['ISBN'])

recommended_linreg["LinReg_Rating"] = loaded_linreg.predict(X)

# объединяем в одну таблицу предсказанные рейтинги
final_rec = pd.merge(svd_rec, recommended_linreg, on="ISBN", how="inner")
final_rec = pd.merge(final_rec, books, on="ISBN", how="inner")
final_rec = pd.merge(final_rec, prepared_raitings, on="ISBN", how="inner")

# сортируем книги по убыванию рейтинга линейной модели
final_rec.sort_values("LinReg_Rating", ascending=False, inplace=True)

"""
Title: Goodnight Moon Board Book
Author: Margaret Wise Brown
SVD_Rating: 8.495695851977084
LinReg_Rating: 7.941563866945958
Average-Book-Rating: 9.714285714285714

Title: Hop on Pop (I Can Read It All by Myself Beginner Books)
Author: Dr. Seuss
SVD_Rating: 8.291104768151898
LinReg_Rating: 7.897416735040417
Average-Book-Rating: 8.833333333333334

Title: The Green Mile
Author: Stephen King
SVD_Rating: 8.203310692802829
LinReg_Rating: 7.853065212809429
Average-Book-Rating: 8.647058823529411

Title: The Magician's Nephew (rack) (Narnia)
Author: C. S. Lewis
SVD_Rating: 8.283761979756003
LinReg_Rating: 7.836665200777319
Average-Book-Rating: 8.67741935483871

Title: All Around the Town
Author: Mary Higgins Clark
SVD_Rating: 8.095774718576658
LinReg_Rating: 7.82662103391578
Average-Book-Rating: 8.196078431372548
"""
