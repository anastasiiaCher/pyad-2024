import pickle
from surprise import SVDpp
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy
import pandas as pd

def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # Убираем значения с рейтингом 0
    df = df[df['Book-Rating'] > 0]

    # Исключаем книги, которые были оценены только один раз
    book_counts = df['ISBN'].value_counts()
    df = df[df['ISBN'].isin(book_counts[book_counts > 1].index)]

    # Исключаем пользователей, которые поставили только одну оценку
    user_counts = df['User-ID'].value_counts()
    df = df[df['User-ID'].isin(user_counts[user_counts > 1].index)]

    # Удаляем строки с пропусками
    df.dropna(inplace=True)

    return df

def modeling(ratings: pd.DataFrame) -> None:

    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=108)

    param_grid = {
        'n_factors': [50, 100, 150],
        'n_epochs': [20, 30, 40],
        'lr_all': [0.002, 0.005, 0.01],
        'reg_all': [0.02, 0.1, 0.2]
    }

    small_data = ratings.sample(frac=0.2, random_state=42)
    small_data = Dataset.load_from_df(small_data[['User-ID', 'ISBN', 'Book-Rating']], reader)

    gs = GridSearchCV(SVDpp, param_grid, measures=['mae'], cv=3, n_jobs=-1)
    gs.fit(small_data)

    best_params = gs.best_params['mae']
    print(f"Лучшие параметры: {best_params}")
    svd = gs.best_estimator['mae']

    svd.fit(trainset)

    predictionss = svd.test(testset)
    maee = accuracy.mae(predictionss)
    print("MAE:", maee)

    with open("svd.pkl", "wb") as file:
        pickle.dump(svd, file)