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