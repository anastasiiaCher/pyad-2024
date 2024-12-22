import pandas as pd
import pickle
from surprise import accuracy
from sklearn.metrics import mean_absolute_error

import linreg_rec

books = pd.read_csv("Books.csv")
ratings = pd.read_csv("Ratings.csv")

books = linreg_rec.books_preprocessing(books)

ratings = linreg_rec.ratings_preprocessing(ratings)

linreg_rec.modeling(books, ratings)

with open('linreg.pkl', 'rb') as f:
    loaded_linreg = pickle.load(f)
td = pd.read_csv("linreg_test.csv")
y = td.pop("y")
predictions = loaded_linreg.predict(td)
mae = mean_absolute_error(y, predictions)
print(mae < 1.5)

from svd_rec import ratings_preprocessing, modeling
ratings = pd.read_csv("Ratings.csv")

ratings = ratings_preprocessing(ratings)

modeling(ratings)

with open('svd.pkl', 'rb') as f:
    loaded_svd = pickle.load(f)
td = pd.read_csv("svd_test.csv")
predictions = loaded_svd.test(td.values)
mae = accuracy.mae(predictions)
print(mae < 1.3)