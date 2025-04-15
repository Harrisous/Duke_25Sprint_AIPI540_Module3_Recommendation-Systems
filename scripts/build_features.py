"""
Script to generate text profiles and embeddings for users and movies
"""
import os
import pathlib
import pandas as pd
import json
import sys
from tqdm import tqdm
from dotenv import load_dotenv
import random
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Define paths
CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = CURRENT_DIR.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Add models directory to path for imports
sys.path.append(str(CURRENT_DIR.parent / "models" / "dnn"))

# Check if we have GOOGLE_API_KEY in environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Import Google Genai if API key is available
if GOOGLE_API_KEY:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("WARNING: GOOGLE_API_KEY not found in environment.")
    print("Using random vectors for embeddings instead of actual LLM embeddings.")

def generate_user_profiles():
    """
    Generate descriptive text profiles for users based on their ratings
    """
    print("Generating user profiles...")
    
    # Load data
    users_file = RAW_DATA_DIR / "ml-100k" / "u.user"
    ratings_file = RAW_DATA_DIR / "ml-100k" / "u.data"
    movies_file = RAW_DATA_DIR / "ml-100k" / "u.item"
    genres_file = RAW_DATA_DIR / "ml-100k" / "u.genre"
    
    # Load users
    users = pd.read_csv(
        users_file, 
        sep="|", 
        names=["user_id", "age", "gender", "occupation", "zip_code"]
    )
    
    # Load ratings
    ratings = pd.read_csv(
        ratings_file, 
        sep="\t", 
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    
    # Load movies with proper encoding
    column_names = ["movie_id", "title", "release_date", "video_release_date", 
                   "IMDb_URL"] + [f"genre_{i}" for i in range(19)]
    movies = pd.read_csv(
        movies_file,
        sep="|",
        names=column_names,
        encoding="latin-1"
    )
    
    # Load genres
    genres = pd.read_csv(
        genres_file,
        sep="|",
        names=["genre", "genre_id"],
        encoding="latin-1"
    )
    genres_dict = dict(zip(genres["genre_id"], genres["genre"]))
    
    # Add genre names to movies
    for i, genre_name in genres_dict.items():
        movies[genre_name] = movies[f"genre_{i}"]
    
    # Get genre list for each movie
    def get_genres(row):
        movie_genres = []
        for genre, has_genre in [(g, row[g]) for g in genres_dict.values()]:
            if has_genre == 1:
                movie_genres.append(genre)
        return ", ".join(movie_genres)
    
    movies["genres"] = movies.apply(get_genres, axis=1)
    
    # Create user profile descriptions
    user_profiles = []
    
    for user_id in tqdm(users["user_id"].unique()):
        # Get user info
        user_info = users[users["user_id"] == user_id].iloc[0]
        
        # Get user ratings
        user_ratings = ratings[ratings["user_id"] == user_id]
        
        # Sort by rating and get top-rated movies
        top_movies = user_ratings.sort_values("rating", ascending=False).head(5)
        
        # Get movie details for top-rated movies
        top_movie_details = []
        for _, movie_rating in top_movies.iterrows():
            movie = movies[movies["movie_id"] == movie_rating["item_id"]].iloc[0]
            top_movie_details.append({
                "title": movie["title"],
                "genres": movie["genres"],
                "rating": movie_rating["rating"]
            })
        
        # Create user profile text
        profile_text = f"Age: {user_info['age']}, gender: {user_info['gender']}, occupation: {user_info['occupation']}. "
        
        if top_movie_details:
            profile_text += "Rating history includes: "
            movie_texts = []
            for movie in top_movie_details:
                movie_text = f"'{movie['title']}' (Genre: {movie['genres']}) rated {movie['rating']} stars"
                movie_texts.append(movie_text)
            profile_text += ", ".join(movie_texts)
        
        # Store user profile
        user_profiles.append({
            "user_id": str(user_id),
            "llm_text": profile_text
        })
    
    # Save user profiles
    with open(PROCESSED_DATA_DIR / "user_profiles.json", "w") as f:
        json.dump(user_profiles, f)
    
    print(f"Generated {len(user_profiles)} user profiles.")
    return user_profiles

def generate_movie_profiles():
    """
    Generate descriptive text profiles for movies
    """
    print("Generating movie profiles...")
    
    # Load movies file
    movies_file = RAW_DATA_DIR / "ml-100k" / "u.item"
    genres_file = RAW_DATA_DIR / "ml-100k" / "u.genre"
    
    # Load movies with proper encoding
    column_names = ["movie_id", "title", "release_date", "video_release_date", 
                   "IMDb_URL"] + [f"genre_{i}" for i in range(19)]
    movies = pd.read_csv(
        movies_file,
        sep="|",
        names=column_names,
        encoding="latin-1"
    )
    
    # Load genres
    genres = pd.read_csv(
        genres_file,
        sep="|",
        names=["genre", "genre_id"],
        encoding="latin-1"
    )
    genres_dict = dict(zip(genres["genre_id"], genres["genre"]))
    
    # Add genre names to movies
    for i, genre_name in genres_dict.items():
        movies[genre_name] = movies[f"genre_{i}"]
    
    # Get genre list for each movie
    def get_genres(row):
        movie_genres = []
        for genre, has_genre in [(g, row[g]) for g in genres_dict.values()]:
            if has_genre == 1:
                movie_genres.append(genre)
        return ", ".join(movie_genres)
    
    movies["genres"] = movies.apply(get_genres, axis=1)
    
    # Create movie profile descriptions
    movie_profiles = []
    
    for _, movie in tqdm(movies.iterrows(), total=len(movies)):
        # Extract year from title
        title = movie["title"]
        year = ""
        if "(" in title and ")" in title:
            year_part = title.split("(")[-1].split(")")[0]
            if year_part.isdigit():
                year = year_part
        
        # Create movie profile text
        profile_text = f"Title: {title}"
        if year:
            profile_text += f", Year: {year}"
        
        profile_text += f", Genres: {movie['genres']}"
        if pd.notna(movie["release_date"]) and movie["release_date"]:
            profile_text += f", Release date: {movie['release_date']}"
        
        # Store movie profile
        movie_profiles.append({
            "item_id": str(movie["movie_id"]),
            "llm_text": profile_text
        })
    
    # Save movie profiles
    with open(PROCESSED_DATA_DIR / "movie_profiles.json", "w") as f:
        json.dump(movie_profiles, f)
    
    print(f"Generated {len(movie_profiles)} movie profiles.")
    return movie_profiles

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60)
)
def get_embedding(text):
    """
    Get embedding for text using Google Gemini API
    """
    if not GOOGLE_API_KEY:
        # Return random embedding if no API key
        return [random.random() * 0.1 for _ in range(768)]
    
    try:
        model = genai.GenerativeModel(
            "models/text-embedding-004"
        )
        result = model.embed_content(text)
        embedding = result.embedding
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise

def generate_embeddings():
    """
    Generate embeddings for user and movie profiles
    """
    print("Generating embeddings for profiles...")
    
    # Load profiles
    try:
        with open(PROCESSED_DATA_DIR / "user_profiles.json", "r") as f:
            user_profiles = json.load(f)
    except FileNotFoundError:
        print("User profiles not found. Run generate_user_profiles first.")
        return
    
    try:
        with open(PROCESSED_DATA_DIR / "movie_profiles.json", "r") as f:
            movie_profiles = json.load(f)
    except FileNotFoundError:
        print("Movie profiles not found. Run generate_movie_profiles first.")
        return
    
    # Generate user embeddings
    print("Generating user embeddings...")
    user_embeddings = []
    
    for user_profile in tqdm(user_profiles):
        embedding = get_embedding(user_profile["llm_text"])
        user_embeddings.append({
            "user_id": user_profile["user_id"],
            "embedding": embedding,
            "llm_text": user_profile["llm_text"]
        })
    
    # Save user embeddings
    with open(PROCESSED_DATA_DIR / "user_embeddings.json", "w") as f:
        json.dump(user_embeddings, f)
    
    # Generate movie embeddings
    print("Generating movie embeddings...")
    movie_embeddings = []
    
    for movie_profile in tqdm(movie_profiles):
        embedding = get_embedding(movie_profile["llm_text"])
        movie_embeddings.append({
            "item_id": movie_profile["item_id"],
            "embedding": embedding,
            "llm_text": movie_profile["llm_text"]
        })
    
    # Save movie embeddings
    with open(PROCESSED_DATA_DIR / "movie_embeddings.json", "w") as f:
        json.dump(movie_embeddings, f)
    
    print("Embeddings generation completed.")

def main():
    """
    Main function to build features
    """
    print("Building features for recommendation system...")
    
    # Generate user profiles
    user_profiles = generate_user_profiles()
    
    # Generate movie profiles
    movie_profiles = generate_movie_profiles()
    
    # Generate embeddings
    generate_embeddings()
    
    print("Feature building completed.")
    print("\nNext steps:")
    print("1. Train the model: python models/dnn/model_v2.py")
    print("2. Start the API server: python models/dnn/api_server.py")

if __name__ == "__main__":
    main()
