import os

import numpy as np


class NaiveRecommendation:
    def __init__(self, is_auto_loaded=True):
        self.user_item_matrix = self._data_loader() if is_auto_loaded else None

    def train(self, user_item_matrix):
        """
        Memorize the user-item interaction matrix.
        :param user_item_matrix: A matrix where rows represent users and columns represent items.
                                 Each entry represents the user's rating for the item (0 if not rated).
        """
        self.user_item_matrix = user_item_matrix  # rating marks matrix

    def recommend(self, user_index=-1, user_vector=None, top_k=5):
        """
        Recommend items to a user based on similar users.
        :param user_index: Index of the user to recommend items for. if user_index = -1, then use user_vector
        :param user_vector: User vector used in cold start
        :param top_k: Number of *items* to recommend.
        :return: List of recommended item indices.
        """

        if user_index != -1 and user_vector == None:
            # existing user
            user_vector = self.user_item_matrix[user_index]
            similarity = self._cosine_similarity(
                user_vector, self.user_item_matrix
            )  # compute similarity, shape = (#user,)
            similarity[user_index] = -1  # exclude self
            most_similar_user = np.argmax(similarity)

            # recommend items that the most similar user has rated but the target user has not
            similar_user_ratings = self.user_item_matrix[most_similar_user]
            user_ratings = self.user_item_matrix[user_index]

        else:
            # cold start -> use user_vector
            is_all_zero = not np.any(user_vector)
            if is_all_zero:
                # if no existing data, then user_vector set to average vector
                user_vector = np.floor(np.mean(self.user_item_matrix, axis=0))

            similarity = self._cosine_similarity(
                user_vector, self.user_item_matrix
            )  # compute similarity, shape = (#user,)
            similarity[user_index] = -1  # exclude self
            most_similar_user = np.argmax(similarity)

            # recommend items that the most similar user has rated but the target user has not
            similar_user_ratings = self.user_item_matrix[most_similar_user]
            user_ratings = user_vector

        # sort recommendations by the similar user's ratings and return the top_k items
        recommendations = np.where((similar_user_ratings > 0) & (user_ratings == 0))[0]
        recommended_items = recommendations[
            np.argsort(similar_user_ratings[recommendations])
        ][::-1]
        return recommended_items[:top_k]

    def _cosine_similarity(self, vector, matrix):
        """
        Compute cosine similarity between a vector and each row in a matrix.
        :param vector: 1D array.
        :param matrix: 2D array.
        :return: 1D array of similarity scores (between users).
        """
        dot_product = np.dot(matrix, vector)
        norm_vector = np.linalg.norm(vector)  # get the norm
        norm_matrix = np.linalg.norm(matrix, axis=1)  # get the norm, axis=1 -> row
        return dot_product / (
            norm_matrix * norm_vector + 1e-10
        )  # Add small value to avoid division by zero

    def _data_loader(self):
        # Dynamically construct the file path relative to the script's location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(
            current_dir, "..", "..", "data", "raw", "ml-100k", "u.data"
        )
        data = np.loadtxt(file_path, delimiter="\t", dtype=int)

        # Determine the matrix size
        num_users = data[:, 0].max()
        num_items = data[:, 1].max()

        # Initialize & summarize data
        ratings_matrix = np.zeros((num_users, num_items), dtype=int)
        for user_id, item_id, rating, _ in data:
            ratings_matrix[user_id - 1, item_id - 1] = (
                rating  # user id and movie id starts from 1
            )

        # Print the shape of the resulting matrix
        print(
            f"Load complete, ratings matrix loaded shape: {ratings_matrix.shape}"
        )  # shape: (943,1682)
        return ratings_matrix


def naive_test():
    """Naive test function for demonstration"""
    # example user-movie interaction matrix
    # rows represent users, columns represent movies, and values represent ratings (0 = not rated)
    user_item_matrix = np.array(
        [[5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [0, 0, 5, 4], [0, 0, 4, 0]]
    )

    recommender = NaiveRecommendation(is_auto_loaded=False)
    recommender.train(user_item_matrix)

    for user_index in range(5):
        recommendations = recommender.recommend(user_index, top_k=2)
        print(f"Recommended items for user {user_index}: {recommendations}")


def random_user_vector(vector_length=1682, valid_size=5):
    vector = np.zeros(vector_length, dtype=int)
    random_indices = np.random.choice(
        vector_length, size=valid_size, replace=False
    )  # random places
    vector[random_indices] = np.random.randint(
        1, 6, size=valid_size
    )  # random rates (0-5) on random places
    return vector


def loaded_naive_test():
    # cold start without info
    user_vector = random_user_vector(valid_size=0)
    recommender = NaiveRecommendation(is_auto_loaded=True)
    recommendations = recommender.recommend(user_vector=user_vector, top_k=5)
    print(f"Recommended items for new user with no info: {recommendations}")

    # cold start with info
    user_vector = random_user_vector(valid_size=5)
    recommendations = recommender.recommend(user_vector=user_vector, top_k=5)
    print(f"Recommended items for new user: {recommendations}")

    # existing user:
    user_index = 10
    recommendations = recommender.recommend(user_index=user_index, top_k=5)
    print(f"Recommended items for user {user_index}: {recommendations}")


if __name__ == "__main__":
    loaded_naive_test()
