"""
Run this script ONCE locally to create a lightweight data file
This eliminates the need for scikit-learn in production
"""
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading original pickle file...")
with open('information.pkl', 'rb') as f:
    df = pickle.load(f)

print(f"Loaded {len(df)} movies")
print(f"Columns: {df.columns.tolist()}")

# Build the similarity matrix
print("Building similarity matrix...")
count_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
vector = count_vectorizer.fit_transform(df["TKN"]).toarray()
cosine_sim = cosine_similarity(vector)

print(f"Similarity matrix shape: {cosine_sim.shape}")

# Create lightweight data structure
data = {
    'titles': df['title'].tolist(),
    'similarity_matrix': cosine_sim.tolist()
}

# Save as compressed pickle
print("Saving optimized data...")
with open('movie_data_light.pkl', 'wb') as f:
    pickle.dump(data, f, protocol=4)

import os
file_size = os.path.getsize('movie_data_light.pkl') / (1024 * 1024)
print(f"✅ Created movie_data_light.pkl ({file_size:.2f} MB)")

if file_size > 200:
    print("\n⚠️  File is still too large for Vercel!")
    print("You'll need to use external storage (GitHub Releases).")
else:
    print("✅ File size is good for Vercel deployment!")