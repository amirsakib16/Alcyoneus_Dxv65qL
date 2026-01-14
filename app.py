from flask import Flask, request, jsonify, make_response, render_template
import pandas as pd
import json
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  


base_dir = os.path.dirname(os.path.abspath(__file__))  
pickle_path = os.path.join(base_dir, "information.pkl")
pickle_path = os.path.abspath(pickle_path)  

try:
    with open(pickle_path, "rb") as file:
        df = pickle.load(file)
except FileNotFoundError:
    raise Exception(f"Could not find the file: {pickle_path}. Please verify its location.")

porterstemmer = PorterStemmer()



def TXTtoLST(data):
    return [porterstemmer.stem(i.lower()) for i in data.split()]




count_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
vector = count_vectorizer.fit_transform(df["TKN"]).toarray()
cosine_sim = cosine_similarity(vector)
# 1. Update your imports at the top


# ... (keep your existing path and model loading code) ...

# 2. Add this route BEFORE your other routes
@app.route("/")
def home():
    return render_template("index.html")

# ... (keep your /test, /recommend, and /movies routes) ...

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"message": "CORS test successful!"})

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    movie_name = data.get("movie")
    movie_name_lower = movie_name.lower()
    
    # Find matching movie titles in the DataFrame (case-insensitive)
    matching_movies = df[df["title"].str.lower().str.contains(movie_name_lower)]
    
    if matching_movies.empty:
        return make_response(jsonify({"error": "Movie not found."}), 404)

    # Get the index of the first matching movie
    movie_index = matching_movies.index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:9]

    recommended_movies = [df.iloc[i[0]]["title"] for i in similarity_scores]
    return jsonify({"recommendations": recommended_movies})
@app.route("/movies", methods=["GET"])
def get_movies():
    movie_titles = df["title"].tolist()  # Extract all movie titles
    return jsonify({"movies": movie_titles})

if __name__ == "__main__":
    app.run(debug=True, port=5000)

