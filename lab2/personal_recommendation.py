import pandas as pd

# 1. Чтение данных
ratings_path = "Ratings.csv"  # Путь к файлу с рейтингами
df = pd.read_csv(ratings_path)

# Проверяем содержимое
print("Первые строки файла Ratings.csv:")
print(df.head())

# 2. Поиск пользователя с максимальным количеством нулевых рейтингов
user_zero_ratings = df[df["Book-Rating"] == 0].groupby("User-ID").size()
target_user = user_zero_ratings.idxmax()  # ID пользователя с максимальным количеством 0

print(f"\nПользователь с максимальным количеством нулевых рейтингов: {target_user}")
print(f"Количество нулевых рейтингов: {user_zero_ratings.max()}")

import pickle

# 1. Загрузка модели SVD

with open("svd.pkl", "rb") as file:
    svd = pickle.load(file)

# 2. Фильтрация книг с рейтингом 0 для целевого пользователя
user_ratings = df[df["User-ID"] == target_user]  # Рейтинги целевого пользователя
zero_rated_books = user_ratings[user_ratings["Book-Rating"] == 0]["ISBN"].tolist()

print(
    f"\nКоличество книг с нулевым рейтингом для пользователя {target_user}: {len(zero_rated_books)}"
)

# 3. Предсказание рейтинга SVD для каждой книги
predictions = []
for isbn in zero_rated_books:
    prediction = svd.predict(
        uid=target_user, iid=isbn
    ).est  # uid: User-ID, iid: ISBN, est: предсказанный рейтинг
    predictions.append((isbn, prediction))

# 4. Оставляем только книги с предсказанным рейтингом >= 8
recommended_books = [(isbn, rating) for isbn, rating in predictions if rating >= 8]

print("\nКниги с предсказанным рейтингом >= 8:")
for isbn, rating in recommended_books:
    print(f"ISBN: {isbn}, Предсказанный рейтинг: {rating:.2f}")
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Пути к данным и моделям
books_path = "Books.csv"
ratings_path = "Ratings.csv"
linreg_model_path = "linreg.pkl"

# Шаг 1. Загрузка данных
books = pd.read_csv(books_path)
ratings = pd.read_csv(ratings_path)
print(recommended_books)
# Предположим, что у нас есть рекомендованные книги из SVD с рейтингом >= 8
recommended_books_svd = recommended_books
recommended_books_isbn = [book[0] for book in recommended_books_svd]

import linreg_rec


books_processed = linreg_rec.books_preprocessing(books)
ratings = linreg_rec.ratings_preprocessing(ratings)
data = pd.merge(ratings, books_processed, on="ISBN", how="inner")
data = data[data["ISBN"].isin(recommended_books_isbn)]
data.to_excel("data.xlsx")
with open("le_author.pkl", "rb") as file:
    le_author = pickle.load(file)
with open("le_publisher.pkl", "rb") as file:
    le_publisher = pickle.load(file)

data["Publisher"] = le_publisher.transform(data["Publisher"])
data["Book-Author"] = le_author.transform(data["Book-Author"])

# Шаг 3. Векторизация названий книг
with open("tfidf.pkl", "rb") as file:
    tfidf = pickle.load(file)
book_titles = tfidf.transform(data["Book-Title"]).toarray()

data.dropna()
# Масштабирование числовых признаков
scaler = StandardScaler()
books_scaled = scaler.fit_transform(
    data[["Book-Author", "Publisher", "Year-Of-Publication"]]
)

# Формируем финальный набор признаков
X_linreg = pd.concat(
    [
        data[["ISBN"]].reset_index(drop=True),
        pd.DataFrame(books_scaled),
        pd.DataFrame(book_titles),
    ],
    axis=1,
)


# Шаг 4. Загрузка модели LinReg
with open(linreg_model_path, "rb") as file:
    linreg_model = pickle.load(file)

# Шаг 5. Предсказание рейтингов
linreg_predictions = linreg_model.predict(X_linreg.iloc[:, 1:])

# Добавляем предсказания в DataFrame
X_linreg["Predicted-Rating"] = linreg_predictions

# Шаг 6. Сортировка книг по убыванию рейтинга
sorted_books = X_linreg.sort_values(by="Predicted-Rating", ascending=False)
# Сопоставляем ISBN с названиями книг
isbn_to_title = data.set_index("ISBN")["Book-Title"].to_dict()

# Вывод результатов
print("\nРекомендованные книги (сортировка по линейной модели):")
for _, row in sorted_books.iterrows():
    isbn = row["ISBN"]
    title = isbn_to_title.get(isbn, "Unknown Title")
    print(
        f"ISBN: {isbn}, Название: {title}, Предсказанный рейтинг: {row['Predicted-Rating']:.2f}"
    )
"""
ISBN: 0345339681, Название: The Hobbit : The Enchanting Prelude to The Lord of the Rings, Предсказанный рейтинг: 9.00
ISBN: 0064400042, Название: On the Banks of Plum Creek, Предсказанный рейтинг: 8.50
ISBN: 059035342X, Название: Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback)), Предсказанный рейтинг: 8.37
ISBN: 0872860175, Название: Howl and Other Poems (Pocket Poets), Предсказанный рейтинг: 8.26
ISBN: 0439064872, Название: Harry Potter and the Chamber of Secrets (Book 2), Предсказанный рейтинг: 8.09
ISBN: 0380725584, Название: The Indian in the Cupboard (rack) (Indian in the Cupboard), Предсказанный рейтинг: 8.06
ISBN: 0064409422, Название: The Lion, the Witch and the Wardrobe (Full-Color Collector's Edition), Предсказанный рейтинг: 8.05
ISBN: 039480029X, Название: Hop on Pop (I Can Read It All by Myself Beginner Books), Предсказанный рейтинг: 8.02
ISBN: 0394900014, Название: Cat in the Hat (I Can Read It All by Myself Beginner Books), Предсказанный рейтинг: 8.01
ISBN: 0440404193, Название: Are You There God?  It's Me, Margaret, Предсказанный рейтинг: 7.98
ISBN: 0440219078, Название: The Giver (21st Century Reference), Предсказанный рейтинг: 7.96
ISBN: 0671032658, Название: The Green Mile, Предсказанный рейтинг: 7.95
ISBN: 0440406498, Название: The Black Cauldron (Chronicles of Prydain (Paperback)), Предсказанный рейтинг: 7.95
ISBN: 0380702843, Название: The Return of the Indian (Indian in the Cupboard), Предсказанный рейтинг: 7.92
ISBN: 0671617028, Название: The Color Purple, Предсказанный рейтинг: 7.86
ISBN: 0694003611, Название: Goodnight Moon Board Book, Предсказанный рейтинг: 7.85
ISBN: 0836220854, Название: Far Side Gallery 2, Предсказанный рейтинг: 7.82
ISBN: 0064400557, Название: Charlotte's Web (Trophy Newbery), Предсказанный рейтинг: 7.78
ISBN: 0440940001, Название: Island of the Blue Dolphins (Laurel Leaf Books), Предсказанный рейтинг: 7.78
ISBN: 0064401847, Название: Bridge to Terabithia, Предсказанный рейтинг: 7.77
ISBN: 088166247X, Название: The Very Best Baby Name Book in the Whole Wide World, Предсказанный рейтинг: 7.77
ISBN: 055321246X, Название: Walden and Other Writings, Предсказанный рейтинг: 7.67
ISBN: 0060508302, Название: Angels Everywhere: A Season of Angels/Touched by Angels (Avon Romance), Предсказанный рейтинг: 7.66
ISBN: 0440235502, Название: October Sky: A Memoir, Предсказанный рейтинг: 7.65
ISBN: 0425083837, Название: The Hunt for Red October, Предсказанный рейтинг: 7.63
ISBN: 0440998050, Название: A Wrinkle in Time, Предсказанный рейтинг: 7.59
ISBN: 0440498058, Название: A Wrinkle In Time, Предсказанный рейтинг: 7.58
ISBN: 0689818769, Название: Frindle, Предсказанный рейтинг: 7.56
ISBN: 0425105334, Название: The Talisman, Предсказанный рейтинг: 7.55
ISBN: 0836218353, Название: Yukon Ho!, Предсказанный рейтинг: 7.55
ISBN: 0345361792, Название: A Prayer for Owen Meany, Предсказанный рейтинг: 7.55
ISBN: 0590464639, Название: The Biggest Pumpkin Ever, Предсказанный рейтинг: 7.55
ISBN: 0440202043, Название: Shell Seekers, Предсказанный рейтинг: 7.55
ISBN: 0553211374, Название: Persuasion, Предсказанный рейтинг: 7.55
ISBN: 0064471101, Название: The Magician's Nephew (rack) (Narnia), Предсказанный рейтинг: 7.55
ISBN: 0064405052, Название: The Magician's Nephew (Narnia), Предсказанный рейтинг: 7.54
ISBN: 0060808934, Название: Skinwalkers, Предсказанный рейтинг: 7.54
ISBN: 0440967694, Название: The Outsiders, Предсказанный рейтинг: 7.54
ISBN: 0451628047, Название: Inferno (Mentor), Предсказанный рейтинг: 7.54
ISBN: 0743400526, Название: She Said Yes : The Unlikely Martyrdom of Cassie Bernall, Предсказанный рейтинг: 7.54
ISBN: 0060915544, Название: The Bean Trees, Предсказанный рейтинг: 7.54
ISBN: 067168390X, Название: Lonesome Dove, Предсказанный рейтинг: 7.53
ISBN: 0394820371, Название: The Phantom Tollbooth, Предсказанный рейтинг: 7.51
ISBN: 0440901588, Название: A Swiftly Tilting Planet, Предсказанный рейтинг: 7.50
ISBN: 0679457526, Название: Into Thin Air : A Personal Account of the Mount Everest Disaster, Предсказанный рейтинг: 7.49
ISBN: 0307010368, Название: Snow White and the Seven Dwarfs, Предсказанный рейтинг: 7.48
ISBN: 0553272535, Название: Night, Предсказанный рейтинг: 7.45
ISBN: 1551667193, Название: 311 Pelican Court, Предсказанный рейтинг: 7.42
ISBN: 0449206440, Название: The Crystal Cave, Предсказанный рейтинг: 7.42
ISBN: 0920668372, Название: Love You Forever, Предсказанный рейтинг: 7.37
"""
