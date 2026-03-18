import streamlit as st
import pickle
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from core.extractor import FeatureExtractor

# --- Page Config ---
st.set_page_config(page_title="Ulos Motif Search", layout="wide")
st.title("🧶 Ulos Motif Visual Search Engine")
st.write("Upload an image of a traditional textile, and our AI will find the most visually similar motifs in the archive.")

# --- Load Data & Model ---
# We use st.cache_resource so the model and data only load once, not on every click
@st.cache_resource
def load_system():
    extractor = FeatureExtractor()
    with open("embeddings.pkl", "rb") as f:
        database = pickle.load(f)
    return extractor, database

try:
    extractor, database = load_system()
except FileNotFoundError:
    st.error("Error: embeddings.pkl not found. Please run indexer.py first!")
    st.stop()

# Extract all vectors from the database into a single numpy matrix for fast comparison
db_vectors = np.array([item["vector"] for item in database])
db_paths = [item["path"] for item in database]

# --- UI: File Uploader ---
uploaded_file = st.file_uploader("Upload a query image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    query_img = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Your Query")
        st.image(query_img, use_container_width=True)
        
    with st.spinner("Analyzing motif and searching archive..."):
        # 1. Extract feature vector of the uploaded image
        query_vector = extractor.extract(query_img)
        
        # 2. Calculate Cosine Similarity against the entire database
        # Reshape query to a 2D array as expected by scikit-learn
        similarities = cosine_similarity(query_vector.reshape(1, -1), db_vectors)[0]
        
        # 3. Get the top 4 most similar indices
        top_k_indices = np.argsort(similarities)[::-1][:4]
        
    # --- UI: Display Results ---
    st.divider()
    st.subheader("Top Matches found in Archive")
    
    # Create a grid of 4 columns
    cols = st.columns(4)
    for i, idx in enumerate(top_k_indices):
        match_path = db_paths[idx]
        match_score = similarities[idx]
        
        match_img = Image.open(match_path)
        with cols[i]:
            st.image(match_img, use_container_width=True)
            st.caption(f"Similarity Score: {match_score:.2f}")