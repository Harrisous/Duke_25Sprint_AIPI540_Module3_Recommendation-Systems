import json
import pandas as pd
import os
import pathlib

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = CURRENT_DIR / ".." / ".." / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Load data
ratings = pd.read_csv(RAW_DATA_DIR / "ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
users = pd.read_csv(RAW_DATA_DIR / "ml-100k/u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"])

# Movie genres
genre_df = pd.read_csv(RAW_DATA_DIR / "ml-100k/u.genre", sep="|", names=["genre", "genre_id"], index_col=False)
genres = genre_df["genre"].tolist()

# Movie information
movie_columns = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + genres
movies = pd.read_csv(RAW_DATA_DIR / "ml-100k/u.item", sep="|", encoding="latin-1", header=None,
                     names=movie_columns, usecols=range(5 + len(genres)))

user_id_map = json.load(open(DATA_DIR / "processed" / "user_id_map.json"))
item_id_map = json.load(open(DATA_DIR / "processed" / "item_id_map.json"))

# Generate genre labels for each movie
def extract_genres(row):
    return [genre for genre in genres if row[genre] == 1]

movies["genres"] = movies.apply(extract_genres, axis=1)

# Movie Text Format
movies["llm_text"] = movies.apply(
    lambda row: f"Movie '{row['title']}', release date: {row['release_date']}. Genres: {', '.join(row['genres'])}.",
    axis=1
)

# Construct movie title lookup table (for assembling user rating history)
movie_title_map = dict(zip(movies["item_id"], movies["title"]))

# Construct rating history lookup table (for assembling user rating history)
merged = ratings.merge(movies[["item_id", "title", "genres"]], on="item_id")
user_history = merged.groupby("user_id").apply(
    lambda x: [f"'{row['title']}' (Genre: {', '.join(row['genres'])}) rated {int(row['rating'])} stars" for _, row in x.iterrows()]
).to_dict()

# User text format
def make_user_text(row):
    uid = row["user_id"]
    age = row["age"]
    gender = row["gender"]
    occupation = row["occupation"]
    history = user_history.get(uid, [])
    history_text = "; ".join(history[:10]) if history else "No rating history"
    return f"Age: {age}, gender: {gender}, occupation: {occupation}. Rating history includes: {history_text}."

users["llm_text"] = users.apply(make_user_text, axis=1)

# Save user and movie texts
user_texts = users[["user_id", "llm_text"]].copy()  # Create a copy instead of a view
movie_texts = movies[["item_id", "llm_text"]].copy()  # Create a copy instead of a view

# Using loc for assignment
user_texts.loc[:, "user_idx"] = user_texts["user_id"].map(lambda x: user_id_map[str(x)])
movie_texts.loc[:, "item_idx"] = movie_texts["item_id"].map(lambda x: item_id_map[str(x)])

# Sort user_texts by user_idx in ascending order
user_texts = user_texts.sort_values(by="user_idx", ascending=True)
movie_texts = movie_texts.sort_values(by="item_idx", ascending=True)

user_texts.to_csv(DATA_DIR / "processed" / "user_texts_for_llm.csv", index=False)
movie_texts.to_csv(DATA_DIR / "processed" / "movie_texts_for_llm.csv", index=False)
