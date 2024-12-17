import pandas as pd
# 2. Загрузка данных (исходный датасет)
ratings = pd.read_csv('/content/Ratings.csv')  # Замените на путь к вашему файлу с рейтингами
books = pd.read_csv('/content/Books.csv')  # Замените на путь к вашему файлу с книгами

# Исправляем строку 209538
books.loc[209538, "Book-Title"] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
books.loc[209538, "Book-Author"] = "Michael Teitelbaum"
books.loc[209538, "Year-Of-Publication"] = "2000"
books.loc[209538, "Publisher"] = "DK Publishing Inc"

# Исправляем строку 220731
books.loc[220731, "Book-Title"] = "Peuple du ciel, suivi de 'Les Bergers'"
books.loc[220731, "Book-Author"] = "Jean-Marie Gustave Le Clézio"
books.loc[220731, "Year-Of-Publication"] = "2003"
books.loc[220731, "Publisher"] = "Gallimard"

# Исправляем строку 221678
books.loc[221678, "Book-Title"] = "DK Readers: Creating the X-Men, How Comic Book Artists Work (Level 4: Proficient Readers)"
books.loc[221678, "Book-Author"] = "Michael Teitelbaum"
books.loc[221678, "Year-Of-Publication"] = "2000"
books.loc[221678, "Publisher"] = "DK Publishing Inc"

from datetime import datetime

# Получим текущий год
current_year = datetime.now().year

# Фильтрация данных
books = books[(books["Year-Of-Publication"].astype(str).str.isdigit())]  # Оставляем только числовые значения
books["Year-Of-Publication"] = books["Year-Of-Publication"].astype(int)  # Приводим к целым числам
books = books[books["Year-Of-Publication"] <= current_year]  # Удаляем будущие годы

# Проверим результат
books["Year-Of-Publication"].describe()

# Удаление пропусков в критически важных столбцах
books = books.dropna(subset=["Book-Author", "Publisher"])

# Удаление столбцов с картинками
books = books.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"])

# Проверим результат
books.info()

# 1. Загружаем ваши модели SVD и SGDRegressor
svd_model = joblib.load('svd_model.pkl')
sgd_model = joblib.load('sgd_regressor_model.pkl')

# 3. Находим пользователя с наибольшим количеством нулевых рейтингов
zero_ratings_user = ratings[ratings['Book-Rating'] == 0]['User-ID'].value_counts().idxmax()

# 4. Находим книги, которым этот пользователь поставил рейтинг 0
user_zero_books = ratings[(ratings['User-ID'] == zero_ratings_user) & (ratings['Book-Rating'] == 0)]['ISBN']

# 5. Предсказания с помощью SVD для этих книг
svd_predictions = []
for isbn in user_zero_books:
    pred = svd_model.predict(zero_ratings_user, isbn)
    if pred.est >= 8:  # Отбираем книги, для которых предсказанный рейтинг >= 8
        svd_predictions.append((isbn, pred.est))

# Преобразуем в DataFrame
svd_predictions_df = pd.DataFrame(svd_predictions, columns=["ISBN", "Predicted-Rating"])

# Получаем средний рейтинг для каждой книги
book_avg_ratings = ratings.groupby("ISBN")["Book-Rating"].mean()

# Объединяем информацию о книгах с их рейтингами
merged_books = books[books["ISBN"].isin(book_avg_ratings.index)]
merged_books["Average-Rating"] = merged_books["ISBN"].map(book_avg_ratings)

# Проверяем итоговый DataFrame
merged_books = merged_books.dropna(subset=["Average-Rating", "Book-Author", "Publisher", "Year-Of-Publication", "Book-Title"])

# Инициализируем векторизатор для названий книг
tfidf = TfidfVectorizer(stop_words="english", max_features=500)

# Векторизуем названия книг
X_title = tfidf.fit_transform(merged_books["Book-Title"])

# Преобразуем столбцы "Book-Author", "Publisher", "Year-Of-Publication"
encoder_author = LabelEncoder()
merged_books["Author-Encoded"] = encoder_author.fit_transform(merged_books["Book-Author"])

encoder_publisher = LabelEncoder()
merged_books["Publisher-Encoded"] = encoder_publisher.fit_transform(merged_books["Publisher"])

merged_books["Year-Encoded"] = merged_books["Year-Of-Publication"].astype(int)

# Фильтруем merged_books по ISBN из предсказаний SVD
merged_books_filtered = merged_books[merged_books["ISBN"].isin(svd_predictions_df["ISBN"])]

# Векторизуем только те книги, которые были предсказаны SVD
X_title_filtered = tfidf.transform(merged_books_filtered["Book-Title"])

# Объединяем все признаки в одну матрицу
X_scaled = sp.hstack([X_title_filtered, merged_books_filtered[["Author-Encoded", "Publisher-Encoded", "Year-Encoded"]].values])

# Предсказания линейной регрессии
y_pred_regressor = sgd_model.predict(X_scaled)

# Добавляем предсказания линейной модели в DataFrame
svd_predictions_df["LinReg-Predicted-Rating"] = y_pred_regressor

# 7. Сортируем по убыванию предсказанных рейтингов линейной модели
recommendations = svd_predictions_df.sort_values(by="LinReg-Predicted-Rating", ascending=False)

# 8. Выводим топ-10 рекомендаций
top_recommendations = recommendations.head(10)

# Печатаем результат
print(top_recommendations)

#     ISBN             Predicted-Rating     LinReg-Predicted-Rating
#132  0440967694          8.374856              1574.199310
#185  0553272535          9.121778              1561.155928
#84   0394517482          8.015214              1558.800296
#264  0743418174          8.063893              1541.025526
#88   0394832922          8.006827              1526.538323
#39   0307231038          8.200591              1497.376173
#85   039480029X          8.531149              1497.094632
#116  0440224055          8.244291              1496.068264
#164  0515087491          8.058471              1487.734565
#261  0717283534          8.130109              1487.468264
