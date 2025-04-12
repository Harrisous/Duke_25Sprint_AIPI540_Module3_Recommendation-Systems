import numpy as np
import pandas as pd
import os
import pathlib
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
CURRENT_DIR = pathlib.Path().resolve() # Assuming running from the same level as the previous scripts
# Adjust DATA_DIR if your script structure is different
#C:\Users\harsh\OneDrive\Desktop\DL Project\Movie Project\Duke_25Sprint_AIPI540_Module3_Recommendation-Systems\data
DATA_DIR = CURRENT_DIR / "Duke_25Sprint_AIPI540_Module3_Recommendation-Systems" / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
MOVIELENS_DIR = RAW_DATA_DIR / "ml-100k"
movies_file_path = MOVIELENS_DIR / "u.item"

class NaiveRecommendation:
    def __init__(self):
        self.user_item_matrix = None

    def train(self, user_item_matrix):
        """
        Memorize the user-item interaction matrix.
        :param user_item_matrix: A matrix where rows represent users and columns represent items.
                                 Each entry represents the user's rating for the item (0 if not rated).
        """
        self.user_item_matrix = user_item_matrix # rating marks matrix

    def recommend(self, user_index, top_k=5):
        """
        Recommend items to a user based on similar users.
        :param user_index: Index of the user to recommend items for.
        :param top_k: Number of *items* to recommend.
        :return: List of recommended item indices.
        """
        user_vector = self.user_item_matrix[user_index]
        similarity = self._cosine_similarity(user_vector, self.user_item_matrix) # compute similarity, shape = (#user,)
        similarity[user_index] = -1 # exclude self
        most_similar_user = np.argmax(similarity)

        # recommend items that the most similar user has rated but the target user has not
        similar_user_ratings = self.user_item_matrix[most_similar_user]
        user_ratings = self.user_item_matrix[user_index]
        recommendations = np.where((similar_user_ratings > 0) & (user_ratings == 0))[0]

        # sort recommendations by the similar user's ratings and return the top_k items
        recommended_items = recommendations[np.argsort(similar_user_ratings[recommendations])][::-1]
        return recommended_items[:top_k]

    def _cosine_similarity(self, vector, matrix):
        """
        Compute cosine similarity between a vector and each row in a matrix.
        :param vector: 1D array.
        :param matrix: 2D array.
        :return: 1D array of similarity scores (between users).
        """
        dot_product = np.dot(matrix, vector)
        norm_vector = np.linalg.norm(vector) # get the norm
        norm_matrix = np.linalg.norm(matrix, axis=1) # get the norm, axis=1 -> row
        return dot_product / (norm_matrix * norm_vector + 1e-10)  # Add small value to avoid division by zero

class EmbeddingRecommendation:
    def __init__(self):
        """
        Loads pre-computed user embeddings and rating data.
        """
        print("Loading data...")
        try:
            # Load user embeddings (JSON stores list of dicts)
            user_embeddings_list = pd.read_json(PROCESSED_DATA_DIR / "user_embeddings.json", orient='records').to_dict('records')
            # Convert to a dictionary {user_id: embedding_vector} for easy lookup
            self.user_embeddings_dict = {user['user_id']: np.array(user['embedding']) for user in user_embeddings_list}
            self.user_ids = list(self.user_embeddings_dict.keys())
            # Create a matrix of embeddings for faster similarity calculation
            # Ensure the order matches self.user_ids
            self.user_embedding_matrix = np.array([self.user_embeddings_dict[uid] for uid in self.user_ids])

            # Load original ratings data to know what users rated
            rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
            self.ratings_df = pd.read_csv(MOVIELENS_DIR / "u.data", sep='\t', names=rating_cols, engine='python')

            print(f"Loaded embeddings for {len(self.user_ids)} users.")
            print(f"Loaded {len(self.ratings_df)} ratings.")

        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Make sure 'get_text_profiles.py' and 'get_profile_embeddings.py' ran successfully.")
            raise

    def _find_most_similar_user(self, target_user_id):
        """
        Finds the user with the most similar embedding to the target user.
        """
        if target_user_id not in self.user_embeddings_dict:
            raise ValueError(f"User ID {target_user_id} not found in embeddings.")

        target_embedding = self.user_embeddings_dict[target_user_id].reshape(1, -1)

        # Calculate cosine similarity between target user and all users
        similarities = cosine_similarity(target_embedding, self.user_embedding_matrix)[0] # Get the 1D array

        # Find the index of the target user to exclude self-similarity
        target_user_index = self.user_ids.index(target_user_id)
        similarities[target_user_index] = -1 # Set self-similarity to a low value

        # Find the index of the most similar user (excluding self)
        most_similar_user_index = np.argmax(similarities)
        most_similar_user_id = self.user_ids[most_similar_user_index]

        return most_similar_user_id

    def recommend(self, target_user_id, top_k=5, rating_threshold=4):
        """
        Recommend items to a user based on the most similar user found via embeddings.
        :param target_user_id: ID of the user to recommend items for.
        :param top_k: Number of items to recommend.
        :param rating_threshold: Minimum rating for an item to be considered 'liked' by the similar user.
        :return: List of recommended item IDs.
        """
        # 1. Find the most similar user based on text embeddings
        try:
            most_similar_user_id = self._find_most_similar_user(target_user_id)
            print(f"Most similar user to {target_user_id} is {most_similar_user_id}")
        except ValueError as e:
            print(e)
            return []

        # 2. Get items rated highly by the most similar user
        similar_user_ratings = self.ratings_df[
            (self.ratings_df['user_id'] == most_similar_user_id) &
            (self.ratings_df['rating'] >= rating_threshold)
        ]
        similar_user_liked_items = set(similar_user_ratings['item_id'])

        if not similar_user_liked_items:
            print(f"Similar user {most_similar_user_id} has no liked items (rating >= {rating_threshold}).")
            return []

        # 3. Get items already rated by the target user
        target_user_rated_items = set(self.ratings_df[self.ratings_df['user_id'] == target_user_id]['item_id'])

        # 4. Find items liked by similar user but not rated by target user
        items_to_recommend = list(similar_user_liked_items - target_user_rated_items)

        if not items_to_recommend:
            print(f"Target user {target_user_id} has already rated all items liked by similar user {most_similar_user_id}.")
            return []

        # 5. Sort recommendations by the similar user's rating (descending) and return top_k
        # Get the ratings the similar user gave ONLY to the potential recommendations
        recommendation_ratings = similar_user_ratings[similar_user_ratings['item_id'].isin(items_to_recommend)]
        recommendation_ratings = recommendation_ratings.sort_values(by='rating', ascending=False)

        return recommendation_ratings['item_id'].tolist()[:top_k]


if __name__ == "__main__":
    # example user-movie interaction matrix
    # rows represent users, columns represent movies, and values represent ratings (0 = not rated)
    user_item_matrix = np.array([
    [5, 4, 1, 0],  # Alice
    [5, 2, 0, 3],  # Bob
    [5, 0, 0, 4],  # Carol
    [5, 5, 0, 4],  # David
    [5, 4, 0, 3],  # Eve
    [0, 3, 4, 5],  # Frank (new user)
    [0, 0, 5, 4],  # Grace (new user)
    ])

 #--------------------------------------------------Naive Method--------------------------------------------------------------------------
    
    print("Naive Method:")
    recommender = NaiveRecommendation()
    recommender.train(user_item_matrix)

    for user_index in range(user_item_matrix.shape[0]):
        recommendations = recommender.recommend(user_index, top_k=2)
        print(f"Recommended items for user {user_index}: {recommendations}")

#------------------------------------------------Included Embeddings----------------------------------------------------------------------------
    print("Recommendations using embeddings:")
    try:
        Erecommender = EmbeddingRecommendation()

        # --- Load Movie Titles ---
        # Define path to u.item using the MOVIELENS_DIR variable
        movies_file_path = MOVIELENS_DIR / "u.item"
        try:
            movie_cols = ['item_id', 'title']
            # Load with index_col='item_id' for easy lookup
            movies_df = pd.read_csv(
                movies_file_path,
                sep='|',
                encoding='latin-1',
                header=None,
                usecols=[0, 1], # Only load item_id and title
                names=movie_cols,
                index_col='item_id' # Use item_id as the DataFrame index
            )
            print(f"\nLoaded movie titles from {movies_file_path}")
            titles_loaded = True
        except FileNotFoundError:
            print(f"\nError: Movie titles file not found at {movies_file_path}")
            print("Cannot display movie titles.")
            titles_loaded = False
        except Exception as e:
            print(f"\nError loading movie titles: {e}")
            titles_loaded = False
        # --- End Load Movie Titles ---


        # --- Get Recommendations for User 1 ---
        user_id_to_recommend = 1
        print(f"\n--- Recommendations for User {user_id_to_recommend} ---")
        recommendations_ids = Erecommender.recommend(user_id_to_recommend, top_k=10)

        if recommendations_ids:
            print(f"Recommended item IDs: {recommendations_ids}")
            if titles_loaded:
                try:
                    # Look up titles using .loc with the list of IDs
                    recommended_titles = movies_df.loc[recommendations_ids]['title'].tolist()
                    print("\nRecommended movie titles:")
                    for i, title in enumerate(recommended_titles):
                        print(f"- {title} (ID: {recommendations_ids[i]})")
                except KeyError as e:
                    print(f"\nError: Could not find title for one or more item IDs: {e}. Some IDs might be missing from u.item.")
                except Exception as e:
                     print(f"\nAn error occurred during title lookup: {e}")
        else:
            print(f"No recommendations found for user {user_id_to_recommend}.")


        # --- Get Recommendations for User 50 ---
        user_id_to_recommend = 50
        print(f"\n--- Recommendations for User {user_id_to_recommend} ---")
        recommendations_ids = Erecommender.recommend(user_id_to_recommend, top_k=10)

        if recommendations_ids:
            print(f"Recommended item IDs: {recommendations_ids}")
            if titles_loaded:
                try:
                    # Look up titles using .loc
                    recommended_titles = movies_df.loc[recommendations_ids]['title'].tolist()
                    print("\nRecommended movie titles:")
                    for i, title in enumerate(recommended_titles):
                        print(f"- {title} (ID: {recommendations_ids[i]})")
                except KeyError as e:
                    print(f"\nError: Could not find title for one or more item IDs: {e}.")
                except Exception as e:
                     print(f"\nAn error occurred during title lookup: {e}")
        else:
            print(f"No recommendations found for user {user_id_to_recommend}.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
