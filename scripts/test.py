import json
import numpy as np

with open("data/processed/movie_embeddings_openai.json", "r") as f:
    data = json.load(f)
    print(f"Movie embedding dimensions: {len(data[0]['embedding'])}")