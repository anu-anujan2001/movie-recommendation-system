import pandas as pd
import streamlit as st
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- UI ----------------
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="centered")
st.title("🎬 Movie Recommendation System")

# ---------------- Load data ----------------
movies_data = pd.read_csv("movies.csv")

# If your dataset doesn't have title column exactly, fix here
movies_data["title"] = movies_data["title"].fillna("")

selected_features = ["genres", "keywords", "tagline", "cast", "director"]
for feature in selected_features:
    if feature in movies_data.columns:
        movies_data[feature] = movies_data[feature].fillna("")
    else:
        movies_data[feature] = ""  # if column missing, create empty column

# Combine features
combined_features = (
    movies_data["genres"] + " " +
    movies_data["keywords"] + " " +
    movies_data["tagline"] + " " +
    movies_data["cast"] + " " +
    movies_data["director"]
)

# ---------------- Vectorize + Similarity ----------------
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)

# ---------------- Input ----------------
movie_name = st.text_input("Enter your favorite movie name")

list_of_all_titles = movies_data["title"].tolist()

if movie_name.strip() == "":
    st.warning("Please enter your favorite movie name.")
else:
    # Find closest title
    find_close_match = difflib.get_close_matches(
        movie_name, list_of_all_titles, n=5, cutoff=0.4
    )

    if len(find_close_match) == 0:
        st.error("No close match found. Try another movie name (check spelling).")
    else:
        close_match = find_close_match[0]
        st.success(f"Closest match: {close_match}")

        # Get index of matched movie (DataFrame row index)
        index_of_the_movie = movies_data[movies_data["title"] == close_match].index[0]

        # Similarity scores for that movie
        similarity_score = list(enumerate(similarity[index_of_the_movie]))

        # Sort movies by similarity
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        # Collect top 30 recommendations (skip itself)
        recommended_movies = []
        for idx, score in sorted_similar_movies:
            title = movies_data.iloc[idx]["title"]
            if title != close_match:
                recommended_movies.append(title)
            if len(recommended_movies) == 30:
                break

        st.subheader("✅ Movies suggested for you")
        selected_movie = st.selectbox("Pick a recommended movie:", recommended_movies)

        st.write("You selected:", selected_movie)