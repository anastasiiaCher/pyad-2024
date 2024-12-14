import pandas as pd
import numpy as np
import joblib
from surprise import Dataset, Reader

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

# Загрузка сохраненных моделей
model_svd = joblib.load("svd_model.pkl")
linreg = joblib.load("linreg_model.pkl")
scaler = joblib.load("scaler.pkl")
vectorizer = joblib.load("vectorizer.pkl")
authors_columns = joblib.load("authors_columns.pkl")
publishers_columns = joblib.load("publishers_columns.pkl")

# Загрузка данных
ratings = pd.read_csv("Ratings.csv")
books = pd.read_csv("Books.csv", low_memory=False)

# Обработка данных как в основном ноутбуке
books["Year-Of-Publication"] = pd.to_numeric(books["Year-Of-Publication"], errors='coerce')
books = books[(books["Year-Of-Publication"] > 0) & (books["Year-Of-Publication"] <= 2024)].copy()
books = books.dropna(subset=["Book-Author", "Publisher"]).copy()

cols_to_drop = ["Image-URL-S", "Image-URL-M", "Image-URL-L"]
books = books.drop(columns=[c for c in cols_to_drop if c in books.columns], errors="ignore").copy()

# Находим пользователя с наибольшим количеством нулевых рейтингов
zero_ratings = ratings[ratings["Book-Rating"] == 0]
user_zero_counts = zero_ratings["User-ID"].value_counts()
top_user = user_zero_counts.idxmax()

# Книги, которым пользователь поставил 0
top_user_zeros = zero_ratings[zero_ratings["User-ID"] == top_user]["ISBN"].unique()

# Предсказание SVD для этих книг
svd_predictions = []
for isbn in top_user_zeros:
    pred = model_svd.predict(uid=top_user, iid=isbn)
    svd_predictions.append((isbn, pred.est))

svd_predictions = pd.DataFrame(svd_predictions, columns=["ISBN", "svd_pred"])

# Оставляем книги с предсказанным рейтингом >= 8
svd_good = svd_predictions[svd_predictions["svd_pred"] >= 8]

if svd_good.empty:
    print("Рекомендация: Книг для данного пользователя не найдено.")
else:
    # Для этих книг сделаем предсказание линейной модели.
    # Для этого нам нужны данные как при обучении линейной регрессии.
    # Загрузим табличку рейтингов для вычисления среднего рейтинга
    train_ratings = ratings[ratings["Book-Rating"] > 0].copy()
    book_counts = train_ratings["ISBN"].value_counts()
    valid_books = book_counts[book_counts > 1].index
    user_counts = train_ratings["User-ID"].value_counts()
    valid_users = user_counts[user_counts > 1].index

    train_ratings = train_ratings[train_ratings["ISBN"].isin(valid_books)]
    train_ratings = train_ratings[train_ratings["User-ID"].isin(valid_users)]

    # Отфильтруем таблицу books под ISBN, что остались
    books_filtered = books[books["ISBN"].isin(train_ratings["ISBN"].unique())]

    # Вычислим средний рейтинг для каждой книги
    book_mean_ratings = train_ratings.groupby("ISBN")["Book-Rating"].mean().reset_index(name="mean_rating")
    books_merge = pd.merge(books_filtered, book_mean_ratings, on="ISBN", how="inner")

    # Оставим в books_merge только нужные книги из svd_good
    recommend_books = books_merge[books_merge["ISBN"].isin(svd_good["ISBN"])].copy()

    if recommend_books.empty:
        print("Рекомендация: Не удалось найти данные о книгах, удовлетворяющих условиям.")
    else:
        # Функция для подготовки признаков так же, как при обучении
        def prepare_features(df):
            # Векторизация названий
            titles_vec = vectorizer.transform(df["Book-Title"].fillna(""))

            # One-hot для автора и издателя
            # Создаем разреженный вектор фиктивных признаков
            # Для этого сформируем матрицу с нулями и единицами вручную
            # Для авторов
            authors_mat = np.zeros((len(df), len(authors_columns)))
            author_map = {a: i for i, a in enumerate(authors_columns)}
            for i, a in enumerate(df["Book-Author"].values):
                col = "author_" + a
                if col in author_map:
                    authors_mat[i, author_map[col]] = 1.0

            # Для издателей
            publishers_mat = np.zeros((len(df), len(publishers_columns)))
            publisher_map = {p: i for i, p in enumerate(publishers_columns)}
            for i, p_ in enumerate(df["Publisher"].values):
                col = "pub_" + p_
                if col in publisher_map:
                    publishers_mat[i, publisher_map[col]] = 1.0

            # Масштабирование года
            years = df["Year-Of-Publication"].values.reshape(-1, 1)
            years_scaled = scaler.transform(years)

            # Объединяем
            from scipy.sparse import csr_matrix, hstack
            authors_sparse = csr_matrix(authors_mat)
            publishers_sparse = csr_matrix(publishers_mat)
            years_sparse = csr_matrix(years_scaled)

            X = hstack([titles_vec, authors_sparse, publishers_sparse, years_sparse], format='csr')
            return X


        X_rec = prepare_features(recommend_books)
        lin_pred = linreg.predict(X_rec)
        recommend_books["lin_pred"] = lin_pred

        # Подмержим предсказание svd, чтобы иметь информацию для сортировки
        recommend_books = pd.merge(recommend_books, svd_good, on="ISBN", how="inner")

        # Сортируем по убыванию линейной модели
        recommendation = recommend_books.sort_values("lin_pred", ascending=False)

        # Вывод рекомендаций
        print(f"Рекомендация для пользователя с наибольшим количеством нулевых рейтингов (User-ID: {top_user}):")
        print("=" * 80)

        top_recommendations = recommendation[
            ["ISBN", "Book-Title", "Book-Author", "Publisher", "lin_pred", "svd_pred"]
        ].head()

        for idx, row in top_recommendations.iterrows():
            print(f"{idx + 1}. Название книги: {row['Book-Title']}")
            print(f"   Автор: {row['Book-Author']}")
            print(f"   Издатель: {row['Publisher']}")
            print(f"   ISBN: {row['ISBN']}")
            print(f"   Предсказанный рейтинг (LinReg): {row['lin_pred']:.2f}")
            print(f"   Предсказанный рейтинг (SVD): {row['svd_pred']:.2f}")
            print("-" * 80)

"""
Рекомендация для пользователя с наибольшим количеством нулевых рейтингов (User-ID: 198711):
================================================================================
29. Название книги: The Lion, the Witch and the Wardrobe (Full-Color Collector's Edition)
   Автор: C. S. Lewis
   Издатель: HarperTrophy
   ISBN: 0064409422
   Предсказанный рейтинг (LinReg): 9.36
   Предсказанный рейтинг (SVD): 8.24
--------------------------------------------------------------------------------
12. Название книги: The Hobbit : The Enchanting Prelude to The Lord of the Rings
   Автор: J.R.R. TOLKIEN
   Издатель: Del Rey
   ISBN: 0345339681
   Предсказанный рейтинг (LinReg): 8.73
   Предсказанный рейтинг (SVD): 8.10
--------------------------------------------------------------------------------
15. Название книги: Cat in the Hat (I Can Read It All by Myself Beginner Books)
   Автор: Seuss
   Издатель: Random House Books for Young Readers
   ISBN: 0394900014
   Предсказанный рейтинг (LinReg): 8.72
   Предсказанный рейтинг (SVD): 8.07
--------------------------------------------------------------------------------
13. Название книги: Charlotte's Web (Trophy Newbery)
   Автор: E. B. White
   Издатель: HarperTrophy
   ISBN: 0064400557
   Предсказанный рейтинг (LinReg): 8.69
   Предсказанный рейтинг (SVD): 8.44
--------------------------------------------------------------------------------
30. Название книги: By the Shores of Silver Lake (Little House)
   Автор: Laura Ingalls Wilder
   Издатель: HarperTrophy
   ISBN: 0064400050
   Предсказанный рейтинг (LinReg): 8.48
   Предсказанный рейтинг (SVD): 8.09
--------------------------------------------------------------------------------
"""
