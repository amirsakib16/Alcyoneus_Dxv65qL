"""
Run this script ONCE locally to create an ultra-lightweight data file
Stores only top 50 recommendations per movie instead of full matrix
"""
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading original pickle file...")
with open('information.pkl', 'rb') as f:
    df = pickle.load(f)

print(f"Loaded {len(df)} movies")

# Build the similarity matrix
print("Building similarity matrix...")
count_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
vector = count_vectorizer.fit_transform(df["TKN"]).toarray()
cosine_sim = cosine_similarity(vector)

print(f"Similarity matrix shape: {cosine_sim.shape}")

# Instead of storing full matrix, store only top K recommendations per movie
TOP_K = 15  # Store top 15 similar movies per movie (reduced to fit GitHub's 100MB limit)
print(f"Extracting top {TOP_K} recommendations per movie...")

recommendations = {}
for idx in range(len(df)):
    # Get similarity scores for this movie
    similarities = cosine_sim[idx]
    
    # Get indices of top K similar movies (excluding the movie itself)
    top_indices = np.argsort(similarities)[::-1][1:TOP_K+1]
    
    # Store indices and scores (use float16 to save space)
    recommendations[idx] = {
        'indices': top_indices.astype(np.int16).tolist(),  # int16 saves space
        'scores': similarities[top_indices].astype(np.float16).tolist()  # float16 saves space
    }
    
    if (idx + 1) % 500 == 0:
        print(f"Processed {idx + 1}/{len(df)} movies...")

# Create lightweight data structure
data = {
    'titles': df['title'].tolist(),
    'recommendations': recommendations
}

# Save as compressed pickle
print("Saving optimized data...")
with open('movie_data_light.pkl', 'wb') as f:
    pickle.dump(data, f, protocol=4)

import os
file_size = os.path.getsize('movie_data_light.pkl') / (1024 * 1024)
print(f"\n✅ Created movie_data_light.pkl ({file_size:.2f} MB)")
print(f"Original matrix size: ~{(len(df) * len(df) * 8) / (1024 * 1024):.2f} MB")
print(f"Optimized size: {file_size:.2f} MB")
print(f"Space savings: {100 * (1 - file_size / ((len(df) * len(df) * 8) / (1024 * 1024))):.1f}%")

if file_size > 200:
    print("\n⚠️  Still too large. Try reducing TOP_K or using external storage.")
else:
    print("✅ File size is good for deployment!")