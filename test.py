import requests


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

    # Print the returned result
    if response.status_code == 200:
        print("Response:", response.json())
    else:
        print(
            f"Failed to get recommendations. Status code: {response.status_code}, Response: {response.text}"
        )


make_DNN_recommendation(
    user_id=-1, user_text="I like science-fiction movie", num_recommendations=3
)
