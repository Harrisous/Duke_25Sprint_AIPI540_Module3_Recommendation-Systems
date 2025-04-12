from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import sys
import pathlib

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
sys.path.append(CURRENT_DIR.absolute())

from inference_v2 import get_recommendations, predict, model

app = FastAPI(
    title="Movie Recommendation API",
    description="API for movie recommendations using a hybrid recommendation system",
    version="1.0.0",
    docs_url="/",
)

client = genai.Client(api_key="AIzaSyAonKpVdGZlvwmfRiBd9TkakXpZU95ht34")

@retry(
    stop=stop_after_attempt(10),  # Maximum 10 retries
    wait=wait_exponential(multiplier=2, min=2, max=10),  # Exponential backoff: 2s, 4s, 8s
    retry=retry_if_exception_type(ClientError),  # Only retry on ClientError
    before_sleep=lambda retry_state: print(f"Rate limit hit, waiting {retry_state.next_action.sleep} seconds...")
)
def get_llm_explanation(user_profile: str, movie_profile: str) -> str:
    prompt = f"""
    Give detailed explanation for the following user profile and movie profile:
    User profile: {user_profile}
    Movie profile: {movie_profile}
    
    explanation to user why this movie is a good recommendation for the user. Try to convince the user to watch this movie. All your response should be in one short sentence.
    """
    model = "gemini-2.5-pro-preview-03-25"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    content = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    
    return content.text

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
def get_prediction(request: PredictionRequest):
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations", response_model=List[MovieRecommendation])
def get_user_recommendations(request: RecommendationRequest):
    try:
        recommendations = get_recommendations(
            user_id=request.user_id,
            user_text=request.user_text,
            num_recommendations=request.num_recommendations
        )
        
        return [
            MovieRecommendation(
                movie_id=movie_id,
                predicted_rating=rating,
                description=description,
                explanation=get_llm_explanation(user_profile=request.user_text, movie_profile=description)
            )
            for movie_id, rating, description in recommendations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
