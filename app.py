import torch
import os
try:
    torch.classes.__path__ = []
    #print("Applied torch.classes.__path__ workaround.")
except Exception as e:
    #print(f"Could not apply torch.classes workaround: {e}")
    torch.classes.__path__ = []
import torch.nn as nn
import streamlit as st
from models.naive.naive_model import NaiveRecommendation
import os
import numpy as np
import requests
import pandas as pd
import pathlib


CURRENT_DIR = pathlib.Path().resolve()
DATA_DIR = CURRENT_DIR / "data" if (CURRENT_DIR / "data").exists() else CURRENT_DIR
RAW_DATA_DIR = DATA_DIR / "raw"
MOVIELENS_DIR = RAW_DATA_DIR / "ml-100k"
PROCESSED_DATA_DIR = DATA_DIR / "processed" # Define even if not used by AutoRec

move_info_path = MOVIELENS_DIR / "u.item"

def extract_movie_info(file_path):
    '''function to get the movie meta data'''
    movie_dict = {}
    with open(file_path, 'r', encoding='ISO-8859-1') as file:  # Encoding to handle special characters
        for line in file:
            fields = line.strip().split('|')
            if len(fields) >= 5:  # Ensure there are enough fields
                movie_id = fields[0]
                movie_title = fields[1]
                imdb_url = fields[4]
                movie_dict[movie_title] = [movie_id, imdb_url]
    return movie_dict

def get_movie_list(movie_dict):
    '''function to generate the movie list'''
    movie_list = []
    for title, info in movie_dict.items():
        movie_list.append(title)
    return movie_list

def make_naive_cold_recommendation(response_dict, movie_dict):
    '''function to generate the recommendation'''
    naive_recommender = NaiveRecommendation(is_auto_loaded=True)
    movie_num = naive_recommender.user_item_matrix.shape[1]
    user_vector = np.zeros(movie_num, dtype=int)

    # compose user matrix
    for name, rating in response_dict.items():
        if name != "":
            movie_id = int(movie_dict[name][0]) - 1  # convert
            user_vector[movie_id] = rating
    # make recommendation
    recommended_movie_ids = naive_recommender.recommend(user_vector=user_vector)

    # build response
    recommendations = []
    for movie_id in recommended_movie_ids:
        for title, info in movie_dict.items():
            if int(info[0]) == movie_id + 1:  # Convert back to one-based index
                recommendations.append(f"- **{title}**:\n IMDb Link: {info[1]}")
                break

    return "### Recommended Movies:\n" + "\n".join(recommendations)


def make_naive_hot_recommendation(user_index):
    """make user recommendation based on user id (existing user)"""
    recommended_movie_ids = naive_recommender.recommend(user_index=user_index)

    # Build response
    recommendations = []
    for movie_id in recommended_movie_ids:
        for title, info in movie_dict.items():
            if int(info[0]) == movie_id + 1:  # Convert back to one-based index
                recommendations.append(f"- **{title}**:\n IMDb Link: {info[1]}")
                break
    return "### Recommended Movies:\n" + "\n".join(recommendations)


def make_DNN_recommendation(user_id, user_text, num_recommendations):
    """make recommendation based using DNN model"""
    # Define the URL and request body
    url = "http://xiaoquankong.ai:8000/recommendations"
    request_body = {
        "user_id": user_id,
        "user_text": user_text,
        "num_recommendations": num_recommendations,
    }

    # Make the POST request
    response = requests.post(url, json=request_body)

    # Process the returned result
    if response.status_code == 200:
        recommendations = response.json()
        recommendation_text = "### Recommended Movies:\n"
        for rec in recommendations:
            movie_title = rec["description"].split("'")[
                1
            ]  # Extract movie title from description
            description = rec["description"]
            explanation = rec["explanation"]
            recommendation_text += f"- **{movie_title}**\n  {description}\n\n  **Reason**: {explanation}\n\n"
        return recommendation_text
    else:
        return f"Failed to get recommendations. Status code: {response.status_code}, Response: {response.text}"

class AutoRec(nn.Module):
    def __init__(self, num_items, hidden_units=512):
        super(AutoRec, self).__init__()
        self.encoder = nn.Linear(num_items, hidden_units)
        self.activation = nn.ReLU() # Make sure this matches the activation used during training
        self.decoder = nn.Linear(hidden_units, num_items)

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        x = self.decoder(x)
        return x


@st.cache_data 
def load_rating_matrix_and_dims(ratings_file_path):
    try:
        ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
        ratings_df = pd.read_csv(ratings_file_path, sep='\t', names=ratings_cols, engine='python')
        max_user_id = ratings_df['user_id'].max()
        max_item_id = ratings_df['item_id'].max()
        rating_matrix = ratings_df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
        rating_matrix = rating_matrix.reindex(index=range(1, max_user_id + 1), columns=range(1, max_item_id + 1), fill_value=0)
        rating_matrix_np = rating_matrix.values.astype(np.float32)
        print(f"Rating matrix loaded ({rating_matrix_np.shape})") # Feedback in UI
        return rating_matrix_np, max_user_id, max_item_id
    except FileNotFoundError:
        print(f"Ratings file not found at {ratings_file_path}")
        return None, None, None
    except Exception as e:
        print(f"Error loading rating matrix: {e}")
        return None, None, None


@st.cache_resource
def load_trained_autorec_model(num_items, hidden_units, model_state_path='autorec_model_state.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoRec(num_items=num_items, hidden_units=hidden_units)
    try:
        model.load_state_dict(torch.load(model_state_path, map_location=device))
        model.to(device)
        model.eval() # Set to evaluation mode
        print("AutoRec model loaded successfully.") # Feedback in UI
        return model, device
    except FileNotFoundError:
        print(f"Trained AutoRec model state file not found at {model_state_path}")
        return None, None
    except Exception as e:
        print(f"Error loading AutoRec model state: {e}")
        return None, None


def recommend_autorec_pytorch(user_id, model, full_rating_matrix_np, movies_info_df, top_k=10, device='cpu'):
    model.eval()

    if user_id < 1 or user_id > full_rating_matrix_np.shape[0]:
        st.error(f"User ID {user_id} is out of valid range (1-{full_rating_matrix_np.shape[0]})")
        return []

    user_index = user_id - 1

    user_vector_np = full_rating_matrix_np[user_index].reshape(1, -1)
    user_vector_tensor = torch.FloatTensor(user_vector_np).to(device)

    with torch.no_grad():
        predicted_ratings_vector = model(user_vector_tensor)[0]

    predicted_ratings_np = predicted_ratings_vector.cpu().numpy()
    original_ratings_np = user_vector_np[0]

    unrated_indices = np.where(original_ratings_np == 0)[0]

    if len(unrated_indices) == 0:
        st.info(f"User {user_id} has rated all items. Cannot recommend.")
        return []

    predicted_unrated_scores = predicted_ratings_np[unrated_indices]
    valid_predictions_mask = ~np.isnan(predicted_unrated_scores)
    indices_to_sort = unrated_indices[valid_predictions_mask]
    scores_to_sort = predicted_unrated_scores[valid_predictions_mask]

    if len(scores_to_sort) == 0:
        st.warning(f"No valid predictions for unrated items for user {user_id}.")
        return []

    initial_fetch_k = top_k + 10
    sorted_indices_local = np.argsort(scores_to_sort)[::-1]

    recommendations = []
    num_added = 0
    for local_idx in sorted_indices_local:
        if num_added >= top_k:
            break

        item_index = indices_to_sort[local_idx]
        item_id = item_index + 1
        score = scores_to_sort[local_idx]

        try:
            title = movies_info_df.loc[item_id]['title']
            if title.strip().lower() == 'unknown':
                continue
            recommendations.append({'item_id': item_id, 'title': title, 'predicted_rating': score})
            num_added += 1
        except KeyError:
            continue
        except Exception as e:
            st.warning(f"Error getting title for {item_id}: {e}. Skipping.")
            continue

    return recommendations


try:
    movie_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
    movies_df = pd.read_csv(
        move_info_path, sep='|', encoding='ISO-8859-1', header=None,
        names=movie_cols, usecols=[0, 1, 2, 3, 4], index_col='item_id'
    )
    titles_loaded_global = True
    print(f"Movie titles loaded from {move_info_path}") # Added success message
except FileNotFoundError:
    print(f"Movie file not found at {move_info_path}. Some features might not work.")
    movies_df = None
    titles_loaded_global = False
except Exception as e:
    print(f"Error loading movie file: {e}")
    movies_df = None
    titles_loaded_global = False


# meta data
move_info_path = os.path.join("data", "raw", "ml-100k", "u.item")
movie_dict = extract_movie_info(move_info_path)
movie_list = get_movie_list(movie_dict)


# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Naive Approach (cold start)",
        "Naive Approach (existing user)",
        "Machine Learning Approach",
        "Deep Learning Approach",
    ],
)

# Page 1: Naive Approach （cold start）
if page == "Naive Approach (cold start)":
    st.title("Naive Approach (cold start)")
    st.write("### Select movies and provide ratings (1-5) as much as you can:")
    naive_image_path = os.path.join("pic", "Naive.jpg")
    st.image(naive_image_path,  use_container_width=True)
    # List of available movie names
    available_movies = movie_list

    # Dropdowns and rating inputs
    selected_movies = []
    ratings = {}
    for i in range(5):  # Four dropdown-rating pairs
        col1, col2 = st.columns([2, 1])  # Two columns: dropdown and rating input
        with col1:
            movie = st.selectbox(
                f"Select Movie {i + 1}",
                options=[""]
                + [m for m in available_movies if m not in selected_movies],
                key=f"movie_{i}",
            )
        with col2:
            rating = st.number_input(
                f"Rating for Movie {i + 1}",
                min_value=0,
                max_value=5,  # 0-5 points
                step=1,
                key=f"rating_{i}",
            )
        selected_movies.append(movie)
        ratings[movie] = rating

    # Buttons for Submit and Refresh
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit"):
            # Prepare the dictionary with movie names as keys and ratings as values
            result = {
                movie: ratings[movie] if movie else 0 for movie in selected_movies
            }

            # Display recommendations
            st.write("### Recommendation:")
            st.write(make_naive_cold_recommendation(result, movie_dict))
    with col2:
        if st.button("Refresh"):
            # Refresh the page by rerunning the script
            st.rerun()


# Page 2: Naive Approach (existing user)
elif page == "Naive Approach (existing user)":
    st.title("Naive Approach (existing user)")
    st.write("### Select a user ID to get recommendations:")
    naive_image_path = os.path.join("pic", "Naive.jpg")
    st.image(naive_image_path,  use_container_width=True)
    # Dropdown for selecting user ID
    naive_recommender = NaiveRecommendation(is_auto_loaded=True)
    user_num = naive_recommender.user_item_matrix.shape[0]
    user_id = st.selectbox(
        "Select User ID",
        options=list(range(1, user_num + 1)),  # User IDs from 0 to 942
        key="user_id",
    )

    # Buttons for Submit and Refresh
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit"):
            # Display recommendations
            st.write("### Recommendation:")
            st.write(make_naive_hot_recommendation(user_index=user_id-1))
    with col2:
        if st.button("Refresh"):
            # Refresh the page by rerunning the script
            st.rerun()         

# Page 3: Machine Learning Approach
elif page == "Machine Learning Approach":
    st.title("Machine Learning Approach (AutoRec)")
    st.write("Recommendations based on learned user preferences from ratings.")
    for i in range(2):
        ML_image_path = os.path.join("pic", f"ML_p{i+1}.jpg")
        st.image(ML_image_path,  use_container_width=True)
    # Load rating matrix and model
    # Define path to ratings data
    ratings_file_path = os.path.join("data", "raw", "ml-100k", "u.data")
    rating_matrix_np, max_user_id, max_item_id = load_rating_matrix_and_dims(ratings_file_path)

    if rating_matrix_np is not None and titles_loaded_global:
        # Define hyperparameters used during training (MUST MATCH)
        HIDDEN_UNITS = 256 # Example, use the value from your training script
        actual_num_items = rating_matrix_np.shape[1] # Number of items from the matrix

        # Load the model
        autorec_model, device = load_trained_autorec_model(actual_num_items, HIDDEN_UNITS)

        if autorec_model is not None:
            st.write("### Select a user ID to get recommendations:")
            # Dropdown for selecting user ID
            user_id = st.selectbox(
                "Select User ID",
                options=list(range(1, max_user_id + 1)), # User IDs from 1 to max_user_id
                key="autorec_user_id_selector"
            )

            if st.button("Get AutoRec Recommendations"):
                with st.spinner("Generating AutoRec recommendations..."):
                    # Pass the globally loaded movies_df
                    recommendations = recommend_autorec_pytorch(
                        user_id, autorec_model, rating_matrix_np, movies_df, top_k=10, device=device
                    )

                if recommendations:
                    st.write("### Top Recommendations from AutoRec:")
                    results_text = ""
                    for i, rec in enumerate(recommendations):
                        imdb_link = movies_df.loc[rec['item_id']]['imdb_url'] if 'imdb_url' in movies_df.columns else "N/A"
                        results_text += f"{i+1}. **{rec['title']}** (ID: {rec['item_id']})\n"
                        results_text += f"   - Predicted Rating: {rec['predicted_rating']:.4f}\n"
                        if imdb_link != "N/A":
                            results_text += f"   - [IMDb Link]({imdb_link})\n"
                        results_text += "\n"
                    st.markdown(results_text) # Use markdown for links
                else:
                    st.info(f"No recommendations generated for User {user_id}.")
        else:
            st.warning("AutoRec model could not be loaded. Cannot generate recommendations.")
    else:
        st.error("Could not load necessary data (ratings matrix or movie titles) for AutoRec.")

# Page 4: Deep Learning Approach
elif page == "Deep Learning Approach":
    st.title("Deep Learning Approach")
    st.write("### Provide your information to get recommendations:")
    for i in range(3):
        NN_image_path = os.path.join("pic", f"NN_p{i+1}.jpg")
        st.image(NN_image_path,  use_container_width=True)

    # Selection box to ask if the user has a user_id
    has_user_id = st.selectbox(
        "Do you have a user ID?", options=["Yes", "No"], key="has_user_id"
    )

    # Input fields based on the selection
    if has_user_id == "Yes":
        user_id = st.number_input(
            "Enter your User ID:", min_value=1, max_value=180, step=1, key="user_id"
        )
    else:
        user_text = st.text_input(
            "Describe your preferences (e.g., 'I like action movies'):", key="user_text"
        )

    # Buttons for Submit and Refresh
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit"):
            # Call make_DNN_recommendation() and display the result
            if has_user_id == "Yes":
                recommendation = make_DNN_recommendation(
                    user_id=user_id, user_text="", num_recommendations=3
                )
            else:
                recommendation = make_DNN_recommendation(
                    user_id=-1, user_text=user_text, num_recommendations=3
                )

            st.write("### Recommendation:")
            st.write(recommendation)
    with col2:
        if st.button("Refresh"):
            # Clear session state to reset the page
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun()
