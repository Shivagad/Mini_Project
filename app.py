import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    # url = "https://raw.githubusercontent.com/rashida048/Some-NLP-Projects/master/movie_dataset.csv"
    # df = pd.read_csv(url)
    df = pd.read_csv("movie_dataset.csv")
    features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in features:
        df[feature] = df[feature].fillna('')
    df['combined'] = df.apply(lambda row: ' '.join(row[feature] for feature in features), axis=1)
    return df

@st.cache_resource
def build_model(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    feature_matrix = vectorizer.fit_transform(df['combined'])
    similarity = cosine_similarity(feature_matrix)
    return similarity

def recommend(movie_title, df, similarity, n=5):
    if movie_title not in df['title'].values:
        return []
    index = df[df['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(similarity[index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return [df.iloc[i[0]]['title'] for i in sorted_scores]

# STREAMLIT UI
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎥 Movie Recommendation System")
st.markdown("Get similar movie suggestions based on content features like genre, cast, and keywords.")

# Load Data and Build Similarity Matrix
df = load_data()
similarity = build_model(df)

# Dropdown for selecting a movie
movie_list = sorted(df['title'].dropna().unique())
selected_movie = st.selectbox("Choose a movie you like:", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie, df, similarity)
    if recommendations:
        st.success("Top Recommendations:")
        for i, movie in enumerate(recommendations, start=1):
            st.markdown(f"**{i}. {movie}**")
    else:
        st.error("No recommendations found. Try another movie.")
