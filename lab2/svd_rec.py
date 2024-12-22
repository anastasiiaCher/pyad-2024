import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy

def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['Book-Rating'] != 0]
    book_counts = df['ISBN'].value_counts()
    user_counts = df['User-ID'].value_counts()
    valid_books = book_counts[book_counts > 1].index
    valid_users = user_counts[user_counts > 1].index
    df = df[df['ISBN'].isin(valid_books) & df['User-ID'].isin(valid_users)]

    return df

def modeling(ratings: pd.DataFrame) -> None:
    # Шаг 1: разбиение данных на тренировочную и тестовую выборки
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings, reader)

    # Шаг 2: Настройка параметров для GridSearchCV
    param_grid = {
        'n_factors': [50, 100],
        'n_epochs': [20, 30],
        'lr_all': [0.005, 0.01],
        'reg_all': [0.02, 0.1]
    }

    gs = GridSearchCV(SVD, param_grid, measures=['mae'], cv=3)
    gs.fit(data)

    best_params = gs.best_params['mae']
    best_mae = gs.best_score['mae']
    print(f'Лучшие параметры: {best_params}')
    print(f'Лучший MAE: {best_mae}')

    # Шаг 3: обучение модели с лучшими параметрами
    best_svd = gs.best_estimator['mae']
    trainset, testset = train_test_split(data, test_size=0.2)
    best_svd.fit(trainset)

    # Шаг 4: тестирование модели и вывод метрик
    predictions = best_svd.test(testset)
    mae = accuracy.mae(predictions)

    # Шаг 5: сохранение модели
    with open("svd.pkl", "wb") as file:
        pickle.dump(best_svd, file)
