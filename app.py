from flask import Flask, request, jsonify, make_response, render_template
import pandas as pd
import pickle
import os
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variables
df = None
cosine_sim = None

# Simple stemmer to replace NLTK (reduces package size)
def simple_stem(word):
    """Lightweight stemmer without NLTK dependency"""
    word = word.lower()
    suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 'ment', 's']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word

def TXTtoLST(data):
    """Convert text to list of stemmed words"""
    return [simple_stem(word) for word in data.split()]

def load_model_data():
    """Load pickle file from local or remote source"""
    global df, cosine_sim
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(base_dir, "information.pkl")
    
    # Option 1: Try loading from local file first
    if os.path.exists(pickle_path):
        print("Loading from local pickle file...")
        try:
            with open(pickle_path, "rb") as file:
                df = pickle.load(file)
            print(f"Loaded {len(df)} movies from local file")
        except Exception as e:
            print(f"Error loading local file: {e}")
            return False
    else:
        # Option 2: Load from remote URL (GitHub, Vercel Blob, etc.)
        # Uncomment and update URL if using remote storage
        """
        REMOTE_URL = "https://your-storage-url.com/information.pkl"
        print("Loading from remote URL...")
        try:
            response = requests.get(REMOTE_URL, timeout=30)
            response.raise_for_status()
            df = pickle.loads(response.content)
            print(f"Loaded {len(df)} movies from remote URL")
        except Exception as e:
            print(f"Error loading from remote: {e}")
            return False
        """
        print(f"Could not find pickle file at: {pickle_path}")
        return False
    
    # Build similarity matrix
    try:
        if "TKN" not in df.columns:
            print("Error: 'TKN' column not found in dataframe")
            return False
            
        count_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
        vector = count_vectorizer.fit_transform(df["TKN"]).toarray()
        cosine_sim = cosine_similarity(vector)
        print("Similarity matrix built successfully")
        return True
    except Exception as e:
        print(f"Error building similarity matrix: {e}")
        return False

# Load data on startup (compatible with Flask 3.0+)
def initialize():
    """Initialize data before first request"""
    global df, cosine_sim
    if df is None:
        success = load_model_data()
        if not success:
            print("WARNING: Failed to load model data!")

# Call initialize when app starts
with app.app_context():
    initialize()

@app.route("/")
def home():
    """Render home page"""
    return render_template("index.html")

@app.route("/test", methods=["GET"])
def test():
    """Test endpoint to verify API is working"""
    return jsonify({
        "message": "CORS test successful!",
        "status": "running",
        "movies_loaded": len(df) if df is not None else 0
    })

@app.route("/recommend", methods=["POST"])
def recommend():
    """Get movie recommendations based on input movie"""
    if df is None or cosine_sim is None:
        return make_response(jsonify({
            "error": "Model not loaded. Please try again later."
        }), 503)
    
    data = request.get_json()
    
    if not data or "movie" not in data:
        return make_response(jsonify({
            "error": "Please provide a movie name in the request body."
        }), 400)
    
    movie_name = data.get("movie", "").strip()
    
    if not movie_name:
        return make_response(jsonify({
            "error": "Movie name cannot be empty."
        }), 400)
    
    movie_name_lower = movie_name.lower()
    
    # Find matching movie titles (case-insensitive)
    matching_movies = df[df["title"].str.lower().str.contains(movie_name_lower, na=False)]
    
    if matching_movies.empty:
        return make_response(jsonify({
            "error": f"Movie '{movie_name}' not found in database.",
            "suggestion": "Try searching for the movie first using /movies endpoint"
        }), 404)
    
    # Get the index of the first matching movie
    movie_index = matching_movies.index[0]
    exact_title = df.iloc[movie_index]["title"]
    
    # Get similarity scores
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:9]
    
    # Get recommended movie titles
    recommended_movies = [df.iloc[i[0]]["title"] for i in similarity_scores]
    
    return jsonify({
        "query": movie_name,
        "matched": exact_title,
        "recommendations": recommended_movies
    })

@app.route("/movies", methods=["GET"])
def get_movies():
    """Get list of all available movies"""
    if df is None:
        return make_response(jsonify({
            "error": "Movie database not loaded."
        }), 503)
    
    # Optional: Add search/filter functionality
    search_query = request.args.get("search", "").lower()
    
    if search_query:
        filtered_titles = df[df["title"].str.lower().str.contains(search_query, na=False)]["title"].tolist()
        return jsonify({
            "count": len(filtered_titles),
            "movies": filtered_titles
        })
    
    movie_titles = df["title"].tolist()
    return jsonify({
        "count": len(movie_titles),
        "movies": movie_titles
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": df is not None and cosine_sim is not None,
        "total_movies": len(df) if df is not None else 0
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return make_response(jsonify({
        "error": "Endpoint not found"
    }), 404)

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return make_response(jsonify({
        "error": "Internal server error"
    }), 500)

if __name__ == "__main__":
    # Load data if running locally
    if df is None:
        load_model_data()
    app.run(debug=True, port=5000)