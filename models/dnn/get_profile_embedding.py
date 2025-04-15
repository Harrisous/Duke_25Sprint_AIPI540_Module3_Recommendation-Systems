# get profile embedding for each user and each movie

import pandas as pd
import os
import pathlib
import time
from typing import List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = CURRENT_DIR / ".." / ".." / "data"

user_texts = pd.read_csv(DATA_DIR / "processed" / "user_texts_for_llm.csv")
movie_texts = pd.read_csv(DATA_DIR / "processed" / "movie_texts_for_llm.csv")

from google import genai
from google.genai import types
from google.genai.errors import ClientError

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


@retry(
    stop=stop_after_attempt(10),  # Maximum 10 retries
    wait=wait_exponential(
        multiplier=2, min=2, max=60
    ),  # Exponential backoff: 2s, 4s, 8s
    retry=retry_if_exception_type(ClientError),  # Only retry on ClientError
    before_sleep=lambda retry_state: print(
        f"Rate limit hit, waiting {retry_state.next_action.sleep} seconds..."
    ),
)
def get_profile_embedding(text: str) -> List[float]:
    result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=text,
        config=types.EmbedContentConfig(task_type="QUESTION_ANSWERING"),
    )
    return result.embeddings[0].values


def process_with_progress(df: pd.DataFrame, desc: str) -> pd.DataFrame:
    """Process data with progress bar"""
    tqdm.pandas(desc=desc)
    df["embedding"] = df["llm_text"].progress_apply(get_profile_embedding)
    return df


if __name__ == "__main__":
    # get embedding for each user and each movie
    print("Starting processing...")
    user_texts = process_with_progress(user_texts, "Processing user data")
    movie_texts = process_with_progress(movie_texts, "Processing movie data")

    # save embeddings
    print("Saving results...")
    user_texts.to_json(
        DATA_DIR / "processed" / "user_embeddings.json", orient="records"
    )
    movie_texts.to_json(
        DATA_DIR / "processed" / "movie_embeddings.json", orient="records"
    )
    print("Done!")
