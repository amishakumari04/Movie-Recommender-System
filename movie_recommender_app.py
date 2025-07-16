import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load data
@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    return ratings, movies

ratings, movies = load_data()

# Parse genres for filtering if available
all_genres = set()
if 'genres' in movies.columns:
    for g in movies['genres'].dropna():
        for genre in g.split('|'):
            all_genres.add(genre)
    all_genres = sorted(list(all_genres))

# Build matrix
@st.cache_resource
def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

def find_similar_movies(movie_id, X, k, metric='cosine'):
    if movie_id not in movie_mapper:
        return []
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind].reshape(1, -1)
    kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric)
    kNN.fit(X)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)
    similar_ids = [movie_inv_mapper[i] for i in neighbour.flatten() if movie_inv_mapper[i] != movie_id]
    return similar_ids[:k]

def recommend_movies_for_user(user_id, k=10, genre_filter=None):
    try:
        user_id = int(user_id)
    except ValueError:
        return "❌ Invalid User ID"

    df1 = ratings[ratings['userId'] == user_id]
    if df1.empty:
        return f"❌ No ratings found for user ID {user_id}"

    top_movie_id = df1[df1['rating'] == df1['rating'].max()]['movieId'].iloc[0]
    movie_titles = dict(zip(movies['movieId'], movies['title']))

    if top_movie_id not in movie_titles:
        return f"❌ Movie ID {top_movie_id} not found in movies.csv"

    top_movie_title = movie_titles[top_movie_id]
    similar_ids = find_similar_movies(top_movie_id, X, k*3)  # get extra for filtering later

    # Filter by genre if selected
    if genre_filter and 'genres' in movies.columns:
        genre_filtered_ids = []
        movie_genres_map = dict(zip(movies['movieId'], movies['genres']))
        for mid in similar_ids:
            genres = movie_genres_map.get(mid, "")
            if genre_filter in genres.split('|'):
                genre_filtered_ids.append(mid)
            if len(genre_filtered_ids) == k:
                break
        similar_ids = genre_filtered_ids
    else:
        # Just trim to k if no filtering
        similar_ids = similar_ids[:k]

    if not similar_ids:
        return f"😕 No similar movies found matching the genre '{genre_filter}'. Try another."

    result = f"🎬 Since you watched **{top_movie_title}**, you might also like:\n"
    for i in similar_ids:
        result += f"- {movie_titles.get(i, f'Movie ID {i} (title not found)')}\n"
    return result

# ------- Streamlit UI -------

st.title("🎥 Movie Recommender System")

user_input = st.text_input("Enter your User ID", value="1")

# Number of recommendations slider
num_recs = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

# Genre filter dropdown (if genres exist)
genre_filter = None
if all_genres:
    genre_filter = st.selectbox("Filter by Genre (optional)", options=["None"] + all_genres)
    if genre_filter == "None":
        genre_filter = None

if st.button("Recommend Movies"):
    with st.spinner("Finding similar movies..."):
        output = recommend_movies_for_user(user_input, k=num_recs, genre_filter=genre_filter)
    st.markdown(output)
