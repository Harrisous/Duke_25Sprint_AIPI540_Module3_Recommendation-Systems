# get_profile_embeddings_openai.py (renamed for clarity)

import pandas as pd
import os
import pathlib
import time
import numpy as np # Added numpy
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# --- OpenAI Specific Imports ---
from openai import OpenAI, RateLimitError # Use OpenAI client and specific error

# --- Configuration ---
CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
# Make sure these paths correctly point to your data directories
DATA_DIR = CURRENT_DIR / ".." / ".." / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# --- Initialize OpenAI Client ---
# Reads API key from environment variable OPENAI_API_KEY
try:
    client = OpenAI()
    #print("OpenAI client initialized.")
except Exception as e:
    #print(f"Error initializing OpenAI client: {e}")
    #print("Ensure OPENAI_API_KEY environment variable is set.")
    client = None

# Choose an OpenAI embedding model
# text-embedding-3-small is cost-effective and performs well
# text-embedding-3-large offers higher dimensionality (potentially better performance, higher cost)
# text-embedding-ada-002 is an older option
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
#print(f"Using OpenAI embedding model: {OPENAI_EMBEDDING_MODEL}")


# --- Load Data ---
try:
    user_texts = pd.read_csv(PROCESSED_DATA_DIR / "user_texts_for_llm.csv")
    movie_texts = pd.read_csv(PROCESSED_DATA_DIR / "movie_texts_for_llm.csv")
    #print("Loaded user and movie text profiles.")
except FileNotFoundError as e:
     print(f"Error loading text profiles: {e}")
     print("Ensure 'get_text_profiles.py' was run successfully and files are in the correct 'processed' directory.")
     exit()


# --- Modified Embedding Function for OpenAI ---
@retry(
    stop=stop_after_attempt(8),  # Adjust max retries if needed
    wait=wait_exponential(multiplier=2, min=2, max=30), # Exponential backoff
    retry=retry_if_exception_type(RateLimitError),  # Retry specifically on OpenAI RateLimitError
    before_sleep=lambda retry_state: print(f"OpenAI rate limit hit, waiting {retry_state.next_action.sleep:.2f} seconds...")
)
def get_openai_embedding(text: str) -> List[float] | None: # Return type hint updated
    """Generates an embedding using the OpenAI API."""
    if not client:
        #print("OpenAI client not available.")
        return None
    try:
        # OpenAI API might handle newlines poorly in some cases, replace just in case
        text = text.replace("\n", " ")
        # Make the API call [1, 4]
        response = client.embeddings.create(
            input=[text], # Input must be a list, even for single strings
            model=OPENAI_EMBEDDING_MODEL
        )
        # Extract the embedding vector [1, 4, 5]
        embedding = response.data[0].embedding
        return embedding # Returns a list of floats
    except RateLimitError as e:
        # Reraise the error for tenacity to handle the retry
        raise e
    except Exception as e:
        print(f"An error occurred during OpenAI embedding generation for text snippet '{text[:50]}...': {e}")
        return None # Return None on other errors

# --- Processing Function (remains largely the same) ---
def process_with_progress(df: pd.DataFrame, desc: str) -> pd.DataFrame:
    """Process data with progress bar using the OpenAI embedding function"""
    tqdm.pandas(desc=desc)
    # Apply the *new* OpenAI embedding function
    df["embedding"] = df["llm_text"].progress_apply(get_openai_embedding)
    # Optional: Remove rows where embedding generation failed
    initial_count = len(df)
    df = df.dropna(subset=['embedding'])
    if len(df) < initial_count:
        print(f"Warning: Dropped {initial_count - len(df)} rows from '{desc}' due to embedding errors.")
    return df

# --- Main Execution ---
if __name__ == "__main__":
    if client is None:
        #print("Exiting because OpenAI client could not be initialized.")
        exit()

    #print("Starting processing...")
    user_texts_with_embeddings = process_with_progress(user_texts, "Processing user data with OpenAI")
    movie_texts_with_embeddings = process_with_progress(movie_texts, "Processing movie data with OpenAI")

    # Save embeddings (consider adding indent for readability)
    user_output_path = PROCESSED_DATA_DIR / "user_embeddings_openai.json" # Save with a new name
    movie_output_path = PROCESSED_DATA_DIR / "movie_embeddings_openai.json" # Save with a new name

    #print(f"Saving results to {user_output_path} and {movie_output_path}...")
    user_texts_with_embeddings.to_json(user_output_path, orient="records", indent=2)
    movie_texts_with_embeddings.to_json(movie_output_path, orient="records", indent=2)
    #print("Done!")
