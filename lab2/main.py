import pickle

import pandas as pd
from sklearn.metrics import mean_absolute_error

# Загружаем модель и связанные объекты
with open('linreg.pkl', 'rb') as f:
    model_data = pickle.load(f)  # Загружаем словарь с моделью, scaler и tfidf

# Извлекаем модель, scaler и tfidf
loaded_linreg = model_data['model']
tfidf = model_data['tfidf']
scaler = model_data['scaler']

# Загружаем тестовые данные
td = pd.read_csv("linreg_test.csv")

# Преобразуем категориальные признаки
td['Book-Author'] = td['Book-Author'].astype('category').cat.codes
td['Publisher'] = td['Publisher'].astype('category').cat.codes

# Векторизация названий книг в тестовых данных с использованием того же TF-IDF
title_vectors = tfidf.transform(td['Book-Title']).toarray()

# Объединяем все признаки
X_test = pd.concat([
    pd.DataFrame(title_vectors, index=td.index),
    td[['Book-Author', 'Publisher', 'Year-Of-Publication']]
], axis=1)

X_test.columns = X_test.columns.astype(str)

# Нормализация тестовых данных
X_test = scaler.transform(X_test)

# Прогнозируем на тестовых данных
predictions = loaded_linreg.predict(X_test)

# Реальные значения для проверки
y = td['Average-Rating']  # Обратите внимание, что этот столбец должен быть в тестовых данных

# Вычисляем MAE (среднюю абсолютную ошибку)
mae = mean_absolute_error(y, predictions)
print(f'Mean Absolute Error (MAE) on Test Set: {mae}')

# Проверяем, что ошибка меньше или равна 1.5
assert mae <= 1.5, f"Test failed. MAE = {mae}, expected value <= 1.5"

print("Test passed. MAE is within the acceptable range.")
