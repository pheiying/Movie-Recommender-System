import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Load the datasets
links = pd.read_csv('ml-latest-small/links.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
tags = pd.read_csv('ml-latest-small/tags.csv')

# Split the genres and create a binary representation for each genre
genres_dummies = movies['genres'].str.get_dummies(sep='|')
# Add the movieId to the genres dataframe
genres_dummies['movieId'] = movies['movieId']

# Extract year from the title
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$', expand=True)

# Preview the changes
movies.head()

cosine_sim_genres = linear_kernel(genres_dummies.drop('movieId', axis=1), genres_dummies.drop('movieId', axis=1))

# Convert years to decades
movies['decade'] = (movies['year'].astype(float) // 10) * 10

# One-hot encode the decades
decades_dummies = pd.get_dummies(movies['decade'])
decades_dummies['movieId'] = movies['movieId']

# Compute cosine similarity for decades
cosine_sim_decades = linear_kernel(decades_dummies.drop('movieId', axis=1), decades_dummies.drop('movieId', axis=1))

# Create the user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Replace NaN with 0 for the cosine similarity computation
user_item_matrix_filled = user_item_matrix.fillna(0)

# Compute the user-user similarity matrix
user_user_sim_matrix = cosine_similarity(user_item_matrix_filled)
user_user_sim_df = pd.DataFrame(user_user_sim_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

# Compute the item-item similarity matrix
item_item_sim_matrix = cosine_similarity(user_item_matrix_filled.T)
item_item_sim_df = pd.DataFrame(item_item_sim_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)


#Part for NCF
###################################################################################################################
df = pd.read_csv('ml-latest-small/ratings.csv')
movies_df = pd.read_csv('ml-latest-small/movies.csv')

# Encode users and movies as integer indices
user_ids = df["userId"].unique().tolist()
movie_ids = df["movieId"].unique().tolist()

user2user_encoded = {x: i for i, x in enumerate(user_ids)}
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

df["user"] = df["userId"].map(user2user_encoded)
df["movie"] = df["movieId"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie2movie_encoded)

# Split data into train and test
X = df[["user", "movie"]].values
y = df["rating"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

embedding_size = 50

# Model inputs
user_input = layers.Input(shape=(1,))
movie_input = layers.Input(shape=(1,))

# Embeddings
user_embedding = layers.Embedding(num_users, embedding_size)(user_input)
user_vector = layers.Flatten()(user_embedding)

movie_embedding = layers.Embedding(num_movies, embedding_size)(movie_input)
movie_vector = layers.Flatten()(movie_embedding)

# Dot product to compute a prediction
y = layers.Dot(1)([user_vector, movie_vector])

model = keras.Model(inputs=[user_input, movie_input], outputs=y)
model.compile(loss='mse', optimizer='adam')

model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=5, batch_size=32, validation_data=([X_test[:, 0], X_test[:, 1]], y_test))
###################################################################################################################


def get_ncf_movie_recommendations(user_id,num_recommendations=10):
    # Movies the user has already rated
    known_movie_ids = df[df.userId == user_id]["movieId"].tolist()

    # Predict ratings for all movies
    user_array = np.array([user2user_encoded[user_id]] * num_movies)
    movie_array = np.array(range(num_movies))

    predicted_ratings = model.predict([user_array, movie_array])

    # Get movie IDs for the top ratings
    top_movie_indices = predicted_ratings.flatten().argsort()[-num_recommendations:][::-1]
    recommended_movie_ids = [movie_ids[i] for i in top_movie_indices if movie_ids[i] not in known_movie_ids]

    # Convert the movie IDs to movie details
    recommendations_list = []
    for movie_id in recommended_movie_ids:
        movie_row = movies_df[movies_df['movieId'] == movie_id]
        title = movie_row['title'].values[0]
        year = extract_year_from_title(title)
        title_without_year = remove_year_from_title(title)
        tmdb_id = int(links[links['movieId'] == movie_id]['tmdbId'].values[0])
        description, poster_url = get_movie_details(tmdb_id)

        recommendations_list.append({
            'title': title_without_year,
            'description': description,
            'poster_url': poster_url,
            'year': year
        })

    return recommendations_list


def genre_based_scores(user):
    user = int(user)  # Ensure user ID is an integer
    rated_movies = ratings[ratings['userId'] == user]

    # Check if rated_movies is not empty
    if rated_movies.empty:
        raise ValueError(f"No ratings found for user ID {user}")

    # Map movie IDs to indices
    rated_indices = rated_movies['movieId'].apply(lambda x: genres_dummies[genres_dummies['movieId'] == x].index[0])

    # Check if rated_indices is not empty
    if rated_indices.empty:
        raise ValueError(f"No valid indices found for user ID {user}")

    # Calculate predicted scores for all movies based on genre similarity
    predicted_scores = cosine_sim_genres[rated_indices].mean(axis=0)

    scores_series = pd.Series(predicted_scores, index=genres_dummies['movieId'])

    return scores_series


def weighted_genre_based_scores(user):
    # Movies rated by the user with their ratings
    user = int(user)  # Ensure user ID is an integer
    user_ratings = ratings[ratings['userId'] == user]
    rated_movies = user_ratings['movieId'].tolist()
    movie_ratings = user_ratings.set_index('movieId')['rating'].to_dict()

    # Map movie IDs to indices
    rated_indices = [genres_dummies.index[genres_dummies['movieId'] == movie].tolist()[0] for movie in rated_movies]

    # Weighted similarity: Initialize with zeros
    weighted_scores = np.zeros(cosine_sim_genres.shape[0])

    # For each rated movie, add its similarity weighted by the user's rating for that movie
    for idx, movie in zip(rated_indices, rated_movies):
        weighted_scores += cosine_sim_genres[idx] * movie_ratings[movie]

    # Normalize the scores
    weighted_scores /= len(rated_movies)

    # Convert scores to a pandas Series
    movie_scores = pd.Series(weighted_scores, index=genres_dummies['movieId'])

    return movie_scores


def year_based_scores(user):
    # Movies rated by the user
    user = int(user)  # Ensure user ID is an integer
    rated_movies = ratings[ratings['userId'] == user]['movieId'].tolist()

    # Map movie IDs to indices
    rated_indices = [decades_dummies.index[decades_dummies['movieId'] == movie].tolist()[0] for movie in rated_movies]

    # Calculate predicted scores for all movies based on decade similarity
    predicted_scores = np.sum(cosine_sim_decades[rated_indices], axis=0) / len(rated_indices)

    return pd.Series(predicted_scores, index=decades_dummies['movieId'])


def weighted_year_based_scores(user):
    # Movies rated by the user with their ratings
    user = int(user)  # Ensure user ID is an integer
    user_ratings = ratings[ratings['userId'] == user]
    rated_movies = user_ratings['movieId'].tolist()
    movie_ratings = user_ratings.set_index('movieId')['rating'].to_dict()

    # Map movie IDs to indices
    rated_indices = [decades_dummies.index[decades_dummies['movieId'] == movie].tolist()[0] for movie in rated_movies]

    # Weighted similarity: Initialize with zeros
    weighted_scores = np.zeros(cosine_sim_decades.shape[0])

    # For each rated movie, add its similarity weighted by the user's rating for that movie
    for idx, movie in zip(rated_indices, rated_movies):
        weighted_scores += cosine_sim_decades[idx] * movie_ratings[movie]

    # Normalize the scores
    weighted_scores /= len(rated_movies)

    return pd.Series(weighted_scores, index=decades_dummies['movieId'])


def combine_content_based_data(movies, tags):
    # Convert genres into dummy variables
    genres_dummies = movies['genres'].str.get_dummies(sep='|')

    # Extract year from the title and convert to decades
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$', expand=True)
    movies['decade'] = (movies['year'].astype(float) // 10) * 10
    decades_dummies = pd.get_dummies(movies['decade'])

    # Create a TF-IDF vectorizer for the tags
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(tags['tag'])
    tags_df = pd.DataFrame(tfidf_matrix.toarray(), index=tags['movieId'],
                           columns=tfidf_vectorizer.get_feature_names_out())

    # Aggregate the tags for each movie
    agg_tags = tags_df.groupby('movieId').mean()

    # Combine genres, decades, and tags dataframes
    combined_df = movies.join(genres_dummies).join(decades_dummies).merge(agg_tags, on='movieId', how='left').fillna(0)

    # Compute cosine similarity
    cosine_sim_combined = cosine_similarity(combined_df.drop(columns=['movieId', 'title', 'genres', 'year', 'decade']))

    return cosine_sim_combined, combined_df


cosine_sim_combined, combined_df = combine_content_based_data(movies, tags)


def combined_content_based_scores(user, csc=cosine_sim_combined, cd=combined_df):
    # Movies rated by the user with their ratings
    user = int(user)  # Ensure user ID is an integer
    user_ratings = ratings[ratings['userId'] == user]
    rated_movies = user_ratings['movieId'].tolist()
    movie_ratings = user_ratings.set_index('movieId')['rating'].to_dict()

    # Map movie IDs to indices
    movie_indices = pd.Series(cd.index, index=cd['movieId'])
    rated_indices = [movie_indices[movie] for movie in rated_movies]

    # Weighted similarity: Initialize with zeros
    weighted_scores = np.zeros(csc.shape[0])

    # For each rated movie, add its similarity weighted by the user's rating for that movie
    for idx, movie in zip(rated_indices, rated_movies):
        weighted_scores += cosine_sim_combined[idx] * movie_ratings[movie]

    # Normalize the scores
    weighted_scores /= len(rated_movies)

    return pd.Series(weighted_scores, index=combined_df['movieId'])


def user_user_predicted_scores_optimized(user_id, k=50):
    user_id = int(user_id)  # Ensure user ID is an integer
    # Movies rated by the user
    user_rated_movies = user_item_matrix.loc[user_id].dropna().index.tolist()

    # Get the top k similar users for the given user based on threshold
    top_k_users = user_user_sim_df[user_id].nlargest(k + 1).index[1:].tolist()  # +1 to exclude the user itself

    # Global mean rating
    global_mean_rating = ratings['rating'].mean()

    # Predict ratings for movies that the user hasn't rated yet
    predicted_ratings = {}
    for movie in user_item_matrix.columns:
        # Predicted rating is the weighted sum of the ratings from similar users
        similar_users_ratings_series = user_item_matrix.loc[top_k_users, movie].dropna()
        similar_users_ratings = similar_users_ratings_series.values
        weights = user_user_sim_df.loc[similar_users_ratings_series.index, user_id].values

        # Check if the sum of weights is non-zero
        if weights.sum() != 0:
            predicted_rating = np.dot(similar_users_ratings, weights) / weights.sum()
            predicted_ratings[movie] = predicted_rating
        else:
            # Assign the global mean rating as the default score
            predicted_ratings[movie] = global_mean_rating

    return pd.Series(predicted_ratings)


def item_based_predicted_scores(user_id):
    user_id = int(user_id)  # Ensure user ID is an integer
    # Movies that the target user has rated
    user_rated_movies = user_item_matrix.loc[user_id].dropna().index.tolist()

    # Movies that the target user hasn't rated
    user_unrated_movies = [movie for movie in user_item_matrix.columns if movie not in user_rated_movies]

    # Dictionary to store predicted ratings
    predicted_ratings = {}

    # Loop through movies the user hasn't rated
    for movie in user_unrated_movies:
        # Get similar movies based on the similarity matrix
        similar_movies = item_item_sim_df[movie].dropna()
        # Filter to only movies the user has rated
        similar_movies = similar_movies.filter(user_rated_movies)

        # If there are no similar movies, continue to the next movie
        if similar_movies.empty:
            continue

        # Calculate the predicted rating
        numerator = sum(similar_movies * user_item_matrix.loc[user_id][similar_movies.index])
        denominator = sum(similar_movies.abs())

        if denominator == 0:
            continue  # Avoid division by zero

        predicted_rating = numerator / denominator
        predicted_ratings[movie] = predicted_rating

    return pd.Series(predicted_ratings)


def svd_movie_scores(user_id):
    user_id = int(user_id)  # Ensure user ID is an integer
    # Fill NaN values in user_item_matrix with 0
    matrix_filled = user_item_matrix.fillna(0).values

    # Normalize by subtracting mean
    user_ratings_mean = np.mean(matrix_filled, axis=1)
    matrix_demeaned = matrix_filled - user_ratings_mean.reshape(-1, 1)

    # SVD
    U, sigma, Vt = svds(matrix_demeaned, k=50)
    sigma = np.diag(sigma)

    # Predicted ratings
    predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns,
                                        index=user_item_matrix.index)

    # Get user's predicted ratings
    user_ratings = predicted_ratings_df.loc[user_id]

    return user_ratings


def get_movie_recommendations(user, recommender_function, n=10):
    """
    Get the top n movie recommendations for a given user.

    Parameters:
    - user: The user ID
    - recommender_function: The recommender function to use
    - n: Number of top recommendations to return (default is 10)

    Returns:
    - List of dictionaries with movie details including title, poster URL, description, and year
    """
    # Get the scores for all movies using the recommender
    scores = recommender_function(user)

    # Sort movies based on the scores and take top n
    sorted_scores = scores.sort_values(ascending=False).head(n)

    # Fetch movie details for the top n movie IDs
    recommended_movies_df = movies[movies['movieId'].isin(sorted_scores.index)]

    # Convert DataFrame to a list of movie details
    recommendations_list = []
    for _, row in recommended_movies_df.iterrows():
        title = row['title']
        year = extract_year_from_title(title)  # Extract the year from the title
        title_without_year = remove_year_from_title(title)  # Remove the year from the title
        tmdb_id = int(links[links['movieId'] == row['movieId']]['tmdbId'].values[0])  # Get the TMDb ID
        description, poster_url = get_movie_details(tmdb_id)

        recommendations_list.append({
            'title': title_without_year,  # Use the modified title without the year
            'description': description,
            'poster_url': poster_url,
            'year': year
        })

    return recommendations_list


def remove_year_from_title(title):
    """
    Remove the year from a movie title.

    Parameters:
    - title: The movie title

    Returns:
    - The title without the year
    """
    import re
    year_pattern = re.compile(r'\(\d{4}\)$')  # Match the year in parentheses at the end of the title
    return re.sub(year_pattern, '', title).strip()  # Remove the matched year and strip any leading/trailing whitespace


def extract_year_from_title(title):
    """
    Extract the year from a movie title.

    Parameters:
    - title: The movie title

    Returns:
    - The extracted year as a string (e.g., '2009')
    """
    # Search for a 4-digit year pattern in the title
    import re
    year_pattern = re.compile(r'\b\d{4}\b')
    match = year_pattern.search(title)
    if match:
        return match.group()
    else:
        return ''  # Return an empty string if no year is found


def get_movie_details(tmdb_id):
    api_url = "https://api.themoviedb.org/3/movie/{}"
    api_key = "b467b354a2546f1e4440e1f71203e99a"  # Dont SHARE MY API KEY YEA :)

    # Get movie details
    response = requests.get(api_url.format(tmdb_id), params={"api_key": api_key})

    if response.status_code == 200:
        movie_details = response.json()
        # Get movie description and image URL
        description = movie_details.get('overview', 'No description available')
        image_url = "https://image.tmdb.org/t/p/w500" + movie_details.get('poster_path', '')
        return description, image_url
    else:
        return 'Error: Unable to fetch movie details', ''
