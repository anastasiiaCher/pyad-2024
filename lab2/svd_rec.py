import pandas as pd
import pickle
from surprise import SVDpp
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import PredictionImpossible


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df['Book-Rating'] != 0]  

    # Исключаем книги, которые были оценены только один раз
    book_counts = df['ISBN'].value_counts()
    df = df[df['ISBN'].isin(book_counts[book_counts > 1].index)]
    
    # Исключаем пользователей, которые поставили только одну оценку
    user_counts = df['User-ID'].value_counts()
    df = df[df['User-ID'].isin(user_counts[user_counts > 1].index)]

    df.dropna(inplace=True)  # Удаляем строки с пропусками
    return df

from surprise.model_selection import GridSearchCV

def modeling(ratings: pd.DataFrame, test_file: str) -> None:
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

    param_grid = {
    'n_factors': [50, 100],
    'reg_all': [0.02, 0.05],
    'lr_all': [0.005, 0.01],
    'n_epochs': [20, 30],
}
    gs = GridSearchCV(SVDpp, param_grid, measures=['mae'], cv=3)
    gs.fit(data)

    best_params = gs.best_params['mae']
    print(f"Лучшие параметры: {best_params}")
    svd = gs.best_estimator['mae']

    trainset = data.build_full_trainset()
    svd.fit(trainset)

    # Оценка на тестовой выборке
    test_data = pd.read_csv(test_file, header=None, names=['User-ID', 'ISBN', 'Book-Rating'])
    test_data['Book-Rating'] = test_data['Book-Rating'].astype(int)
    reader_test = Reader(rating_scale=(1, 10))
    testset = list(zip(test_data['User-ID'], test_data['ISBN'], test_data['Book-Rating']))

    try:
        predictions = svd.test(testset)
        mae = accuracy.mae(predictions)
        print(f"Mean Absolute Error (MAE) на тестовой выборке: {mae}")
    except PredictionImpossible as e:
        print("Предсказание невозможно для некоторых данных:", e)

    with open("svd.pkl", "wb") as file:
        pickle.dump(svd, file)


if __name__ == "__main__":
    ratings = pd.read_csv("Ratings.csv")

    ratings = ratings_preprocessing(ratings)

    test_file = "svd_test.csv"  # путь к тестовой выборке
    modeling(ratings, test_file)