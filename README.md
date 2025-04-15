# Hybrid Recommendation System
Team members: Haochen Li, Xiaoquan Kong, Harshitha Rasamsetty, Violet Suh

## Project Overview
This project implements an advanced hybrid recommendation system that combines collaborative filtering, content-based filtering, and deep learning techniques to provide personalized movie recommendations to users. The system leverages both user behavior data and descriptive content to create a robust recommendation engine that addresses cold-start problems and provides explainable recommendations.

## Key Features
- **Hybrid Architecture**: Combines collaborative filtering with content-based approaches using deep neural networks
- **LLM-Enhanced Profiles**: Uses LLM embeddings to capture semantic relationships between users and items
- **Cold-Start Handling**: Supports new users through textual descriptions using Gemini AI
- **Explainable Recommendations**: Provides personalized explanations for recommendations using Gemini 2.5 Pro
- **REST API**: FastAPI-based interface for easy integration with frontend applications
- **Cross-Validation**: Rigorous 5-fold cross-validation for model evaluation

## Data Pipeline

### Data Collection
We utilized the MovieLens 100K dataset which contains:
- 100,000 ratings (1-5) from 943 users on 1,682 movies
- Demographic information for users (age, gender, occupation)
- Movie metadata including title, release date, and genres

### Data Preprocessing
1. **Data Cleaning**: Removed inconsistencies and missing values
2. **Feature Engineering**: 
   - Generated rich textual profiles for both users and movies
   - User profiles combine demographic information with movie preferences
   - Movie profiles include title, release year, genres, and plot elements

### Feature Extraction
1. **Text Profile Generation**:
   - For users: Combined demographic data with their top-rated movie preferences
   - For movies: Created descriptive summaries using available metadata
   
2. **Embedding Generation**:
   - Used Google's Gemini API to create dense vector representations (768-dimensional)
   - These embeddings capture semantic meaning of user preferences and movie content
   - Allows for measuring similarity in a shared latent space

### Data Flow Architecture
```
Raw Data → Text Profile Generation → LLM Embedding → Neural Network Processing → Ratings Prediction
```

## Modeling Approaches

### Previous Approaches in Recommendation Systems

#### Naive Approaches
- **Popularity-Based**: Recommends the most popular items to all users
  - *Limitations*: Ignores individual user preferences and niche interests
- **Random Recommendations**: Provides random suggestions
  - *Limitations*: No personalization, poor user experience

#### Non-Deep Learning Approaches
1. **Content-Based Filtering**:
   - Recommends items similar to what a user has liked in the past
   - Uses TF-IDF and cosine similarity on item features
   - *Limitations*: Cannot capture latent relationships, focuses only on content

2. **Collaborative Filtering**:
   - **User-Based**: Finds similar users and recommends what they liked
   - **Item-Based**: Finds similar items to what a user has liked
   - **Matrix Factorization** (e.g., SVD, ALS): Decomposes user-item interaction matrix
   - *Limitations*: Cold-start problem, sparsity issues, difficult to incorporate auxiliary information

3. **Hybrid Methods**:
   - Combines multiple recommendation approaches
   - *Limitations*: Often relies on manual feature engineering and heuristic combinations

#### Deep Learning Approaches
1. **Neural Collaborative Filtering**:
   - Uses neural networks to learn user-item interactions
   - Replaces matrix factorization with deep neural networks
   
2. **Deep Factorization Machines**:
   - Combines factorization machines with deep neural networks
   - Captures both low and high-order feature interactions

3. **Transformer-Based Approaches**:
   - Leverages attention mechanisms to model complex sequential behaviors
   - Examples: BERT4Rec, SASRec

### Our Approach: LLM-Enhanced Hybrid Neural Network

Our model combines the strengths of collaborative filtering and content-based approaches using deep learning:

1. **Embedding Layer**:
   - Pre-trained LLM embeddings capture semantic information about users and items
   - These fixed embeddings provide rich representations of users and items

2. **Neural Processing Layers**:
   - Parallel MLP networks process user and item embeddings separately
   - User MLP: Transforms user embeddings into a latent preference space
   - Item MLP: Transforms item embeddings into the same latent space
   - Architecture: 
     ```
     Input (768) → Dense (1024) → ReLU → Dense (512) → ReLU → Dense (256)
     ```

3. **Similarity Calculation**:
   - Cosine similarity between processed user and item vectors
   - Non-linear mapping to transform similarity scores to rating scale (1-5)

4. **Cold-Start Handling**:
   - For new users: Generate embeddings directly from textual descriptions
   - Fallback mechanisms use average embeddings when no data is available

5. **Explanation Component**:
   - Leverages Gemini 2.5 Pro to generate natural language explanations
   - Customized explanations based on user profiles and movie characteristics

## Evaluation

### Methodology
We evaluated our model using 5-fold cross-validation on the MovieLens 100K dataset with the following metrics:

1. **Root Mean Square Error (RMSE)**:
   - Measures the square root of the average squared differences between predicted and actual ratings
   - Penalizes larger errors more severely

2. **Mean Absolute Error (MAE)**:
   - Measures the average absolute differences between predicted and actual ratings
   - More interpretable as average rating error

### Results

| Metric | Our Model | Matrix Factorization | Neural CF | 
|--------|-----------|----------------------|-----------|
| RMSE   | 0.923     | 0.945                | 0.932     |
| MAE    | 0.728     | 0.749                | 0.736     |

Our model outperforms traditional matrix factorization by 2.3% on RMSE and 2.8% on MAE, and neural collaborative filtering by 1.0% on RMSE and 1.1% on MAE.

### Ablation Studies

We conducted ablation studies to understand the contribution of different components:

| Model Variant | RMSE | MAE |
|---------------|------|-----|
| Full Model | 0.923 | 0.728 |
| Without LLM Embeddings | 0.962 | 0.751 |
| Without MLP Layers | 0.954 | 0.745 |
| Without Non-Linear Rating Mapping | 0.935 | 0.732 |

These results demonstrate that:
- LLM embeddings provide the most significant improvement (4.2% RMSE reduction)
- MLP layers for transformation contribute substantially (3.4% RMSE reduction)
- Non-linear rating mapping offers modest but meaningful gains (1.3% RMSE reduction)

## Practical Implementation

### API Service
We implemented a REST API using FastAPI that provides:
- GET/POST endpoints for predictions and recommendations
- Support for cold-start scenarios through textual descriptions
- Explanation generation for recommendations
- Swagger documentation for easy client integration

### Performance Optimization
- Batch processing for efficient inference
- Caching mechanisms for frequently requested recommendations
- Retry logic with exponential backoff for LLM API calls

## Ethical Considerations

### Privacy Concerns
- **User Data Protection**: Our system processes sensitive user preference data, requiring proper security measures and anonymization techniques
- **Informed Consent**: Users should be informed about how their data is used for recommendations

### Bias and Fairness
- **Popularity Bias**: Recommendation systems can amplify popularity biases, potentially limiting user exposure to diverse content
- **Demographic Biases**: Different demographic groups may receive systematically different recommendation quality
- **Mitigation Efforts**: We implemented regularization techniques and diversity metrics to monitor and reduce bias

### Filter Bubbles and Echo Chambers
- Recommendation systems risk creating filter bubbles by reinforcing existing preferences
- We incorporated diversity measures to ensure users receive varied recommendations

### Transparency and Explainability
- Our system generates natural language explanations to provide transparency
- This helps users understand why certain items are recommended to them

### Environmental Impact
- LLM-based solutions have significant computational requirements
- We optimized our architecture to minimize unnecessary computation and reduce the carbon footprint

## Future Improvements
- Implement A/B testing framework for model comparison
- Add support for more diverse recommendation strategies
- Enhance explainability with more detailed content analysis
- Scale the system to handle larger datasets
- Incorporate time-aware components to capture preference evolution
- Develop advanced bias mitigation techniques

## Presentation Slide:
[canvas slide url](https://www.canva.com/design/DAGkkHsZ_2g/-8bR0jgp6UmJYEcPseuDeA/edit?utm_content=DAGkkHsZ_2g&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

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

## Acknowledgments
- MovieLens dataset from GroupLens Research
- Google Gemini API for LLM capabilities
- PyTorch framework for deep learning implementation
