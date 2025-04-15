from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import sys
import pathlib
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
GEMINI_MODEL_NAME = "gemini-2.5-pro-preview-03-25"

# Add current directory to system path
sys.path.append(CURRENT_DIR.absolute())

# Import model-related functions
from inference_v2 import get_recommendations, predict, model

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="API for movie recommendations using a hybrid recommendation system",
    version="1.0.0",
    docs_url="/",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google GenAI client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type(ClientError),
    before_sleep=lambda retry_state: print(
        f"Rate limit hit, waiting {retry_state.next_action.sleep} seconds..."
    ),
)
def get_llm_explanation(user_profile: str, movie_profile: str) -> str:
    """
    Uses Gemini model to generate a one-sentence persuasive explanation
    for recommending a movie to a user.

    Args:
        user_profile (str): Text description of the user.
        movie_profile (str): Text description of the movie.

    Returns:
        str: One-sentence recommendation explanation.
    """
    prompt = f"""
    Give detailed explanation for the following user profile and movie profile:
    User profile: {user_profile}
    Movie profile: {movie_profile}

    explanation to user why this movie is a good recommendation for the user. Try to convince the user to watch this movie. All your response should be in one short sentence.
    """

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain"
    )

    content = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=contents,
        config=generate_content_config,
    )

    return content.text


# Request/Response Models
class PredictionRequest(BaseModel):
    user_id: int
    item_id: int
    user_text: Optional[str] = None


class RecommendationRequest(BaseModel):
    user_id: int
    user_text: Optional[str] = None
    num_recommendations: Optional[int] = 10


class MovieRecommendation(BaseModel):
    movie_id: int
    predicted_rating: float
    description: str
    explanation: str


@app.post("/predict", response_model=float)
def get_prediction(request: PredictionRequest) -> float:
    """
    Predict the rating of a specific movie for a given user.

    Args:
        request (PredictionRequest): Contains user_id, item_id, and optional user_text.

    Returns:
        float: Predicted rating.
    """
    try:
        if request.user_text:
            user_vec = model.get_fallback_user_with_text(request.user_text)
        else:
            uidx = model.user_id_map.get(request.user_id)
            if uidx is None:
                raise HTTPException(status_code=404, detail="User ID not found")
            user_vec = model.user_llm_emb[uidx].unsqueeze(0)

        prediction = predict(model, user_vec, request.item_id)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction error: " + str(e))


@app.post("/recommendations", response_model=List[MovieRecommendation])
def get_user_recommendations(request: RecommendationRequest) -> List[MovieRecommendation]:
    """
    Generate personalized movie recommendations for a user.

    Args:
        request (RecommendationRequest): Contains user_id, optional user_text,
        and number of recommendations to return.

    Returns:
        List[MovieRecommendation]: List of movie recommendations with explanations.
    """
    try:
        recommendations = get_recommendations(
            user_id=request.user_id,
            user_text=request.user_text,
            num_recommendations=request.num_recommendations,
        )

        return [
            MovieRecommendation(
                movie_id=movie_id,
                predicted_rating=rating,
                description=description,
                explanation=get_llm_explanation(
                    user_profile=request.user_text, movie_profile=description
                ),
            )
            for movie_id, rating, description in recommendations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Recommendation error: " + str(e))


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
