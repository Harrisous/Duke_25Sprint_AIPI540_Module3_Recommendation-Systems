# Hybrid Recommendation System

## Project Overview
This project implements an advanced hybrid recommendation system that combines collaborative filtering, content-based filtering, and deep learning techniques to provide personalized movie recommendations to users. The system leverages both user behavior data and descriptive content to create a robust recommendation engine that addresses cold-start problems and provides explainable recommendations.

## Key Features
- **Hybrid Architecture**: Combines collaborative filtering with content-based approaches using deep neural networks
- **LLM-Enhanced Profiles**: Uses LLM embeddings to capture semantic relationships between users and items
- **Cold-Start Handling**: Supports new users through textual descriptions using Gemini AI
- **Explainable Recommendations**: Provides personalized explanations for recommendations using Gemini 2.5 Pro
- **REST API**: FastAPI-based interface for easy integration with frontend applications
- **Cross-Validation**: Rigorous 5-fold cross-validation for model evaluation

## Technical Architecture

### Data Processing Pipeline
1. **Data Collection**: Uses the MovieLens 100K dataset with user ratings
2. **Feature Extraction**: Processes user and item profiles to generate rich textual descriptions
3. **LLM Embeddings**: Generates embeddings for users and items using advanced language models
4. **Model Training**: Trains a hybrid neural network on the processed data

### Model Architecture
The core recommendation model (`HybridRecModel`) is implemented as a neural network that:
- Processes LLM embeddings for users and items through parallel MLP networks
- Calculates cosine similarity between resulting vectors
- Maps similarity scores to rating predictions using a non-linear transformation

```python
class HybridRecModel(nn.Module):
    def __init__(self, num_users, num_items, id_emb_dim, llm_emb_dim, user_llm_emb, item_llm_emb):
        super().__init__()
        self.user_llm_emb = user_llm_emb  # Fixed, not trainable
        self.item_llm_emb = item_llm_emb
        self.total_dim = llm_emb_dim
        
        # Define neural network layers for processing embeddings
        self.user_mlp = nn.Sequential(...)
        self.item_mlp = nn.Sequential(...)

    def forward(self, user_idx, item_idx):
        # Get embeddings
        user_vec = self.user_llm_emb[user_idx]
        item_vec = self.item_llm_emb[item_idx]
        
        # Process through MLPs
        user_vec = self.user_mlp(user_vec)
        item_vec = self.item_mlp(item_vec)
        
        # Calculate similarity and map to rating range
        similarity = F.cosine_similarity(user_vec, item_vec)
        rating = 1 + 4 * (similarity ** 2)  # non-linear mapping
        return rating
```

### API Service
The recommendation service is exposed through a FastAPI application that provides:
- Prediction endpoint for individual user-item pairs
- Recommendation endpoint for generating top-N recommendations with explanations
- Support for both existing users and cold-start scenarios

## Evaluation Results
The model was evaluated using 5-fold cross-validation on the MovieLens 100K dataset:

| Metric | Value |
|--------|-------|
| RMSE   | 0.923 |
| MAE    | 0.728 |

These results demonstrate strong predictive performance compared to traditional recommendation approaches.

## Setup and Usage

### Requirements
```
requests
pandas
google-genai
tenacity
tqdm
torch
scikit-learn
fastapi
uvicorn
python-dotenv
black
```

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (API keys) in .env file
4. Run the setup script: `python setup.py`

### Running the API Server
```bash
python models/dnn/api_server.py
```

The server will be available at http://localhost:8000 with interactive Swagger documentation.

### Making Recommendations
To get recommendations for existing users:
```python
from models.dnn.inference_v2 import get_recommendations

recommendations = get_recommendations(user_id=1, num_recommendations=10)
```

For new users (cold start):
```python
recommendations = get_recommendations(
    user_id=None, 
    user_text="Age: 33, gender: M, occupation: software engineer. Enjoys sci-fi and action movies.", 
    num_recommendations=10
)
```

## Future Improvements
- Implement A/B testing framework for model comparison
- Add support for more diverse recommendation strategies
- Enhance explainability with more detailed content analysis
- Scale the system to handle larger datasets

## Acknowledgments
- MovieLens dataset from GroupLens Research
- Google Gemini API for LLM capabilities
- PyTorch framework for deep learning implementation
