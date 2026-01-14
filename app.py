from flask import Flask, request, jsonify, make_response, render_template
import pickle
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variables
movie_data = None

def load_model_data():
    """Load pre-computed similarity data from GitHub Releases"""
    global movie_data
    
    # Try local file first (for development)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "movie_data_light.pkl")
    
    if os.path.exists(data_path):
        print("Loading from local file...")
        try:
            with open(data_path, "rb") as file:
                movie_data = pickle.load(file)
            print(f"✅ Loaded {len(movie_data['titles'])} movies from local file")
            return True
        except Exception as e:
            print(f"❌ Error loading local file: {e}")
    
    # Load from GitHub Releases (for production)
    GITHUB_RELEASE_URL = "https://github.com/amirsakib16/Alcyoneus_Dxv65qL/releases/download/v1.0/movie_data_light.pkl"
    
    print(f"Loading from GitHub Releases: {GITHUB_RELEASE_URL}")
    try:
        import urllib.request
        print("Downloading movie data (this may take a moment)...")
        with urllib.request.urlopen(GITHUB_RELEASE_URL, timeout=60) as response:
            movie_data = pickle.loads(response.read())
        print(f"✅ Loaded {len(movie_data['titles'])} movies from GitHub")
        return True
    except Exception as e:
        print(f"❌ Error loading from GitHub: {e}")
        return False

# Initialize data when app starts
def initialize():
    """Initialize data on startup"""
    global movie_data
    if movie_data is None:
        success = load_model_data()
        if not success:
            print("⚠️  WARNING: Failed to load movie data!")

with app.app_context():
    initialize()

@app.route("/")
def home():
    """Render home page"""
    if os.path.exists(os.path.join('templates', 'index.html')):
        return render_template("index.html")
    return jsonify({
        "message": "Movie Recommendation API",
        "endpoints": {
            "/test": "Test API connection",
            "/movies": "Get all movies (GET)",
            "/recommend": "Get recommendations (POST with {movie: 'name'})",
            "/health": "Health check"
        }
    })

@app.route("/test", methods=["GET"])
def test():
    """Test endpoint"""
    return jsonify({
        "message": "API is working!",
        "status": "running",
        "movies_loaded": len(movie_data['titles']) if movie_data else 0
    })

@app.route("/recommend", methods=["POST"])
def recommend():
    """Get movie recommendations"""
    if movie_data is None:
        return make_response(jsonify({
            "error": "Movie data not loaded. Please try again later."
        }), 503)
    
    data = request.get_json()
    
    if not data or "movie" not in data:
        return make_response(jsonify({
            "error": "Please provide a movie name in the request body as {\"movie\": \"name\"}"
        }), 400)
    
    movie_name = data.get("movie", "").strip()
    
    if not movie_name:
        return make_response(jsonify({
            "error": "Movie name cannot be empty."
        }), 400)
    
    movie_name_lower = movie_name.lower()
    titles = movie_data['titles']
    similarity_matrix = movie_data['similarity_matrix']
    
    # Find matching movie (case-insensitive)
    movie_index = None
    exact_title = None
    
    for idx, title in enumerate(titles):
        if movie_name_lower in title.lower():
            movie_index = idx
            exact_title = title
            break
    
    if movie_index is None:
        return make_response(jsonify({
            "error": f"Movie '{movie_name}' not found in database.",
            "suggestion": "Try searching with /movies?search=yourquery"
        }), 404)
    
    # Get similarity scores for this movie
    similarities = similarity_matrix[movie_index]
    
    # Get indices of top 8 similar movies (excluding the movie itself)
    similar_indices = sorted(
        range(len(similarities)), 
        key=lambda i: similarities[i], 
        reverse=True
    )[1:9]
    
    recommended_movies = [titles[i] for i in similar_indices]
    
    return jsonify({
        "query": movie_name,
        "matched": exact_title,
        "recommendations": recommended_movies
    })

@app.route("/movies", methods=["GET"])
def get_movies():
    """Get list of all movies with optional search"""
    if movie_data is None:
        return make_response(jsonify({
            "error": "Movie database not loaded."
        }), 503)
    
    titles = movie_data['titles']
    search_query = request.args.get("search", "").lower()
    
    if search_query:
        filtered_titles = [t for t in titles if search_query in t.lower()]
        return jsonify({
            "count": len(filtered_titles),
            "movies": filtered_titles
        })
    
    return jsonify({
        "count": len(titles),
        "movies": titles
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "data_loaded": movie_data is not None,
        "total_movies": len(movie_data['titles']) if movie_data else 0
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return make_response(jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/test", "/movies", "/recommend", "/health"]
    }), 404)

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return make_response(jsonify({
        "error": "Internal server error"
    }), 500)

if __name__ == "__main__":
    if movie_data is None:
        load_model_data()
    app.run(debug=True, port=5000)