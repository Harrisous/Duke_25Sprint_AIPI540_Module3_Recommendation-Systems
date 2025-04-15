import streamlit as st
from models.naive.naive_model import NaiveRecommendation
import os
import numpy as np
import requests

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
            movie_id = int(movie_dict[name][0]) -1 # convert
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
    '''make user recommendation based on user id (existing user)'''
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
    '''make recommendation based using DNN model'''
    # Define the URL and request body
    url = "http://xiaoquankong.ai:8000/recommendations"
    request_body = {
        "user_id": user_id,
        "user_text": user_text,
        "num_recommendations": num_recommendations
    }

    # Make the POST request
    response = requests.post(url, json=request_body)

    # Process the returned result
    if response.status_code == 200:
        recommendations = response.json()
        recommendation_text = "### Recommended Movies:\n"
        for rec in recommendations:
            movie_title = rec['description'].split("'")[1]  # Extract movie title from description
            description = rec['description']
            explanation = rec['explanation']
            recommendation_text += f"- **{movie_title}**\n  {description}\n\n  **Reason**: {explanation}\n\n"
        return recommendation_text
    else:
        return f"Failed to get recommendations. Status code: {response.status_code}, Response: {response.text}"


# meta data
move_info_path = os.path.join("data", "raw", "ml-100k", "u.item")
movie_dict = extract_movie_info(move_info_path)
movie_list = get_movie_list(movie_dict)



# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Naive Approach (cold start)", "Naive Approach (existing user)", "Machine Learning Approach", "Deep Learning Approach"])

# Page 1: Naive Approach （cold start）
if page == "Naive Approach (cold start)":
    st.title("Naive Approach (cold start)")
    st.write("### Select movies and provide ratings (1-5) as much as you can:")

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
                options=[""] + [m for m in available_movies if m not in selected_movies],
                key=f"movie_{i}"
            )
        with col2:
            rating = st.number_input(
                f"Rating for Movie {i + 1}",
                min_value=0,
                max_value=5, # 0-5 points
                step=1,
                key=f"rating_{i}"
            )
        selected_movies.append(movie)
        ratings[movie] = rating

    # Buttons for Submit and Refresh
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit"):
            # Prepare the dictionary with movie names as keys and ratings as values
            result = {movie: ratings[movie] if movie else 0 for movie in selected_movies}
            
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

    # Dropdown for selecting user ID
    naive_recommender = NaiveRecommendation(is_auto_loaded=True)
    user_num = naive_recommender.user_item_matrix.shape[0]
    user_id = st.selectbox(
        "Select User ID",
        options=list(range(1,user_num+1)),  # User IDs from 0 to 942
        key="user_id"
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
    st.title("Machine Learning Approach")
    st.write("### Enter grades for the following moves:")
    
    # List of move names
    movies = ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"]
    
    # Input boxes for grading
    grades = {}
    for move in movies:
        grades[move] = st.text_input(f"Grade for {move}:", "")

# Page 4: Deep Learning Approach
elif page == "Deep Learning Approach":
    st.title("Deep Learning Approach")
    st.write("### Provide your information to get recommendations:")

    # Selection box to ask if the user has a user_id
    has_user_id = st.selectbox(
        "Do you have a user ID?",
        options=["Yes", "No"],
        key="has_user_id"
    )

    # Input fields based on the selection
    if has_user_id == "Yes":
        user_id = st.number_input(
            "Enter your User ID:",
            min_value=1,
            max_value=180,
            step=1,
            key="user_id"
        )
    else:
        user_text = st.text_input(
            "Describe your preferences (e.g., 'I like action movies'):",
            key="user_text"
        )

    # Buttons for Submit and Refresh
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit"):
            # Call make_DNN_recommendation() and display the result
            if has_user_id == "Yes":
                recommendation = make_DNN_recommendation(user_id=user_id, user_text="", num_recommendations=3)
            else:
                recommendation = make_DNN_recommendation(user_id=-1, user_text=user_text, num_recommendations=3)
            
            st.write("### Recommendation:")
            st.write(recommendation)
    with col2:
        if st.button("Refresh"):
            # Clear session state to reset the page
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun()