import pandas as pd
import pickle
from linreg_rec import modeling, books_preprocessing, ratings_preprocessing

def recomendation(books_data: pd.DataFrame, ratings_data: pd.DataFrame) -> list:
    with open('svd.pkl', 'rb') as svd_file:
        svd_model = pickle.load(svd_file)
    with open('linreg.pkl', 'rb') as linreg_file:
        linreg_model = pickle.load(linreg_file)

    target_user = (
        ratings_data.loc[ratings_data['Book-Rating'] == 0, 'User-ID']
        .value_counts()
        .idxmax()
    )

    user_zero_ratings = ratings_data[(ratings_data['User-ID'] == target_user) & (ratings_data['Book-Rating'] == 0)]

    recommendations_list = []
    feature_columns = [str(i) for i in range(1003)]

    for isbn in user_zero_ratings['ISBN']:
        predicted_rating = svd_model.predict(target_user, isbn).est

        if predicted_rating >= 8:
            book_features = features.loc[merged_data['ISBN'] == isbn].to_numpy()
            book_features_df = pd.DataFrame(book_features, columns=feature_columns)

            linear_prediction = linreg_model.predict(book_features_df)[0]

            recommendations_list.append((isbn, predicted_rating, linear_prediction))

    recommendations_list.sort(key=lambda rec: rec[2], reverse=True)

    return recommendations_list

ratings = pd.read_csv("Ratings.csv")
books = pd.read_csv("Books.csv")

processed_books, processed_ratings = books_preprocessing(books), ratings_preprocessing(ratings)
merged_data, features = modeling(processed_books, processed_ratings)

output_file = "recommendations_for_most_zeroes_user.txt"
with open(output_file, "w") as file:
    for isbn, svd_score, linreg_score in recomendation(processed_books, ratings):
        book_name = books.loc[books['ISBN'] == isbn, 'Book-Title'].values[0]
        file.write(f"Book: {book_name}\nPredicted rating: {svd_score:.2f}\nLinReg rating: {linreg_score:.2f}\n\n")
