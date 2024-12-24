import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv"""

    # Не будем использовать для обучения записи с рейтингом равным 0:
    df_raitings = df[df["Book-Rating"].astype(int) != 0]

    # Будем использовать для обучения алгоритмов те книги, которым оценка поставлена больше min_count раз.
    isbn_to_count = df_raitings['ISBN'].value_counts()
    books_for_processing = isbn_to_count[isbn_to_count > 1].index
    df_raitings = df_raitings[df_raitings['ISBN'].isin(books_for_processing)]

    # Будем использовать для обучения алгоритмов пользователей, которые поставили оценку больше min_count раз.
    user_to_count = df_raitings['User-ID'].value_counts()
    users_for_processing = user_to_count[user_to_count > 1].index
    df_raitings = df_raitings[df_raitings['User-ID'].isin(users_for_processing)]

    return df_raitings


def modeling(ratings: pd.DataFrame) -> None:
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[["User-ID", "ISBN", "Book-Rating"]], reader)

    #1. Разбить данные на тренировочную и обучающую выборки
    trainset, testset = train_test_split(data, test_size=0.3, random_state=42)

    #2. Обучить и протестировать SVD
    #3. Подобрать гиперпараметры
    """ param_grid = {
        'n_factors': [1, 2],
        'lr_all':    [0.006, 0.008],
        'reg_all':   [0.02, 0.04]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['mae'], cv=3, n_jobs=-1, joblib_verbose=0)
    gs.fit(data)

    best_params = gs.best_params['mae']
    print("Best params SVD:", best_params)

    svd = SVD(
        n_factors=best_params['n_factors'],
        lr_all=best_params['lr_all'],
        reg_all=best_params['reg_all'],
        random_state=42
    ) """
    
    svd = SVD()
    
    svd.fit(trainset)
    result = svd.test(testset)
    mae = accuracy.mae(result)

    #4. Сохранить модель
    if mae < 1.3:
        with open("svd.pkl", "wb") as f:
            pickle.dump(svd, f)
        print(f"Модель сохранена. MAE: {mae}")
    else:
        print(f"Модель не сохранена. MAE: {mae}")
