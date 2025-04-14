import numpy as np
import pandas as pd
import os
import pathlib
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI, RateLimitError
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Configuration ---
CURRENT_DIR = pathlib.Path().resolve()
DATA_DIR = CURRENT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
MOVIELENS_DIR = RAW_DATA_DIR / "ml-100k"
movies_file_path = MOVIELENS_DIR / "u.item"



try:
    client = OpenAI()
    #print("client initialized.")
except Exception as e:
    #print(f"Error initializing client: {e}")
    client = None

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small" # or "text-embedding-3-large"

@retry(
    stop=stop_after_attempt(8), #adjust if needed
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(RateLimitError), 
    before_sleep=lambda retry_state: print(f"OpenAI rate limit hit, waiting {retry_state.next_action.sleep:.2f} seconds...")
)
def get_openai_embedding(text: str) -> np.ndarray | None:
    """
    Generates an embedding for the given text using the OpenAI API.
    Returns a numpy array or None if an error occurs.
    """
    if not client:
        #print("OpenAI client not available.")
        return None
    try:
        text = text.replace("\n", " ") # OpenAI API might have issues with newlines
        response = client.embeddings.create(input=[text], model=OPENAI_EMBEDDING_MODEL)
        embedding = response.data[0].embedding
        return np.array(embedding)
    except RateLimitError as e:
        print(f"Caught RateLimitError, tenacity will handle retry: {e}")
        raise 
    except Exception as e:
        print(f"An error occurred during OpenAI embedding generation: {e}")
        return None


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



class EmbeddingRecommender:
    def __init__(self):
        """
        Loads pre-computed movie embeddings and movie metadata.
        """
        #print("Loading pre-computed movie data...")
        try:
            movie_embeddings_list = pd.read_json(PROCESSED_DATA_DIR / "movie_embeddings_openai.json", orient='records').to_dict('records')
            self.movie_embeddings_dict = {movie['item_id']: np.array(movie['embedding']) for movie in movie_embeddings_list}
            self.movie_ids = list(self.movie_embeddings_dict.keys())
            self.movie_embedding_matrix = np.array([self.movie_embeddings_dict[mid] for mid in self.movie_ids])

            movies_file_path = MOVIELENS_DIR / "u.item"
            movie_cols = ['item_id', 'title']
            self.movies_df = pd.read_csv(
                movies_file_path, sep='|', encoding='latin-1', header=None,
                usecols=[0, 1], names=movie_cols, index_col='item_id'
            )
            self.titles_loaded = True
            #print(f"Loaded embeddings for {len(self.movie_ids)} movies.")
            #print(f"Loaded titles for {len(self.movies_df)} movies.")

        except FileNotFoundError as e:
            #print(f"Error loading data: {e}")
            #print("Make sure 'movie_embeddings_openai.json' and 'u.item' exist.")
            #print("Movie embeddings need to be generated first (can use OpenAI or Gemini).")
            self.titles_loaded = False
            raise
        except Exception as e:
             #print(f"An unexpected error occurred during loading: {e}")
             self.titles_loaded = False
             raise


    def _generate_user_profile_text(self, age: int, gender: str, occupation: str, liked_genres: List[str] = None, liked_movies: List[str] = None, disliked_movies: List[str] = None) -> str:
        """
        Creates a descriptive text string from user input.
        """
        profile = f"User profile - Age: {age}, Gender: {gender}, Occupation: {occupation}."
        if liked_genres:
            profile += f" Primarily enjoys genres like: {', '.join(liked_genres)}."
        if liked_movies:
            profile += f" Some favorite movies include: {', '.join(liked_movies)}."
        if disliked_movies:
             profile += f" Dislikes movies like: {', '.join(disliked_movies)}."
        return profile

    def recommend_movies_for_user(self, age: int, gender: str, occupation: str, liked_genres: List[str] = None, liked_movies: List[str] = None, disliked_movies: List[str] = None, top_k=10):
        """
        Generates movie recommendations based on user input profile.
        :param age: User's age
        :param gender: User's gender ('M', 'F', 'O', etc.)
        :param occupation: User's occupation
        :param liked_genres: Optional list of preferred genres
        :param liked_movies: Optional list of liked movie titles (for context)
        :param disliked_movies: Optional list of disliked movie titles (for context/filtering)
        :param top_k: Number of recommendations to return
        :return: List of tuples (movie_title, similarity_score, item_id) or empty list
        """
        #print("\nGenerating profile text...")
        user_profile_text = self._generate_user_profile_text(age, gender, occupation, liked_genres, liked_movies, disliked_movies)
        #print(f"Generated Profile: {user_profile_text}")

        #print("Generating embedding for user profile using OpenAI...")
        user_embedding = get_openai_embedding(user_profile_text)

        if user_embedding is None:
            #print("Failed to generate user embedding. Cannot provide recommendations.")
            return []

        user_embedding = user_embedding.reshape(1, -1)

        #print("Calculating similarity between user profile and all movies...")
        similarities = cosine_similarity(user_embedding, self.movie_embedding_matrix)[0]

        initial_fetch_k = top_k + 5
        top_k_indices = np.argsort(similarities)[-initial_fetch_k:][::-1]

        potential_movie_ids = [self.movie_ids[i] for i in top_k_indices]
        potential_scores = [similarities[i] for i in top_k_indices]


        recommendations = []
        if self.titles_loaded:
            for item_id, score in zip(potential_movie_ids, potential_scores):
                if len(recommendations) >= top_k:
                    break

                try:
                    title = self.movies_df.loc[item_id]['title']

                    if title.strip().lower() == 'unknown':
                        #print(f"Skipping item ID {item_id} because title is 'unknown'.")
                        continue

                    if disliked_movies and title in disliked_movies:
                        #print(f"Skipping disliked movie: {title}")
                        continue

                    recommendations.append((title, score, item_id))

                except KeyError:
                    #print(f"Warning: Movie ID {item_id} not found in title mapping. Skipping.")
                    continue
                except Exception as e:
                    #print(f"Error retrieving title for {item_id}: {e}. Skipping.")
                    continue
        else:
            recommendations = list(zip(potential_movie_ids, potential_scores))[:top_k]

        return recommendations


if __name__ == "__main__":
    # example user-movie interaction matrix
    # rows represent users, columns represent movies, and values represent ratings (0 = not rated)
    user_item_matrix = np.array([
    [5, 4, 1, 0],  # User 1: Alice
    [5, 2, 0, 3],  # User 2: Bob
    [5, 0, 0, 4],  # User 3: Carol
    [5, 5, 0, 4],  # User 4: David
    [5, 4, 0, 3],  # User 5: Eve
    [0, 3, 4, 5],  # User 6: Frank (new user)
    [0, 0, 5, 4],  # User 7: Grace (new user)
    ])

 #--------------------------------------------------Naive Method--------------------------------------------------------------------------
    
    #print("Naive Method:")
    recommender = NaiveRecommendation()
    recommender.train(user_item_matrix)

    for user_index in range(user_item_matrix.shape[0]):
        recommendations = recommender.recommend(user_index, top_k=2)
        #print(f"Recommended items for user {user_index}: {recommendations}")

#------------------------------------------------Included Embeddings----------------------------------------------------------------------------

    try:
        if client is None:
             raise SystemExit("OpenAI client failed to initialize. Check API key/environment variable.")

        recommender = EmbeddingRecommender()

        user_input = {
            "age": 30,
            "gender": "M",
            "occupation": "engineer",
            "liked_genres": ["Sci-Fi", "Action", "Thriller"],
            "liked_movies": ["The Matrix", "Blade Runner"],
            "disliked_movies": ["Titanic"] 
        }

        #print(f"\n--- Generating recommendations for profile: ---")
        #print(user_input)

        recommendations = recommender.recommend_movies_for_user(**user_input, top_k=10)

        if recommendations:
            print("\n--- Top Recommendations ---")
            for i, (title_or_id, score, *rest) in enumerate(recommendations):
                if recommender.titles_loaded: 
                    item_id = rest[0] if rest else 'N/A'
                    print(f"{i+1}. {title_or_id} (ID: {item_id}) - Similarity: {score:.4f}")
                else:
                    print(f"{i+1}. Item ID: {title_or_id} - Similarity: {score:.4f}")


    except FileNotFoundError:
        print("\nExecution failed due to missing files. Please check paths and ensure necessary files exist.")
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main block: {e}")
