# Movie Recommendation System using AutoRec (PyTorch)

## Overview

This project implements a movie recommendation system using the **AutoRec** algorithm, specifically the User-based variant (U-AutoRec), on the MovieLens 100k dataset. AutoRec employs a shallow **autoencoder** neural network, built with **PyTorch**, to learn latent representations (embeddings) of users directly from their rating patterns. It then uses these representations to predict ratings for movies a user hasn't seen and generates recommendations based on these predictions.

This approach differs from methods relying on external embeddings (like OpenAI) by learning user features *endogenously* from the collaborative filtering data itself.

## Methodology: AutoRec

1.  **Shallow Autoencoder Architecture:**
    *   A neural network designed to reconstruct its input.
    *   **Input:** A single user's complete rating vector (length = total number of movies), with 0s representing unrated movies.
    *   **Encoder (Hidden Layer):** A dense layer compresses the input vector into a lower-dimensional latent representation (user embedding). ReLU activation is typically used.
    *   **Decoder (Output Layer):** Another dense layer reconstructs the full rating vector from the latent representation. Linear activation is used to predict rating values.
    *   The network is "shallow" as it typically uses only one hidden layer.

2.  **Training:**
    *   The model is trained to minimize the reconstruction error between the output vector and the input vector.
    *   **Masked Loss Function:** Crucially, the loss (e.g., Mean Squared Error) is calculated *only* on the items the user *has actually rated* (non-zero entries in the input). This ensures the model learns from observed data and doesn't try to force predictions to match the placeholder 0s.
    *   Regularization (L2 / weight decay) is often applied to prevent overfitting.

3.  **Recommendation Generation:**
    *   Input a user's rating vector into the trained model.
    *   The output vector contains predicted ratings for *all* movies, including those originally rated 0.
    *   Identify the items the user hasn't rated (original rating was 0).
    *   Rank these unrated items based on their predicted ratings (highest first).
    *   Return the top-K ranked items as recommendations.
    *   Movies with "unknown" titles in the dataset are optionally skipped.

## Dataset

*   **MovieLens 100k:**
    *   Contains 100,000 ratings (scale 1-5).
    *   From 943 users on 1,682 movies.
    *   Includes user demographics and movie metadata (used here mainly for title lookup).
*   Downloaded automatically by the script or expected in `data/raw/ml-100k/`.
*   Source: [https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/)

## Setup

1.  **Clone the Repository:**
    ```
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Install Dependencies:**
    Create a `requirements.txt` file (or add to an existing one):
    ```
    torch
    pandas
    numpy
    requests
    matplotlib
    ```
    Then install:
    ```
    pip install -r requirements.txt
    ```

3.  **Dataset:**
    *   Run the main script (`ML_model.py`). It includes code to automatically download and extract the MovieLens 100k dataset to `data/raw/ml-100k/` if it's not found.
    *   Alternatively, manually place the extracted contents (`u.data`, `u.item`, etc.) in that directory.

## Usage

1.  **Run the Main Script:**
    *   Execute the Python script containing the AutoRec implementation in the scripts folder `ML_model.py`).
    *   ```
        python scripts/ML_model.py
        ```
    *   The script will perform the following steps:
        1.  Check for/download the dataset.
        2.  Load and prepare the user-item rating matrix.
        3.  Build the AutoRec model.
        4.  Train the model using the masked loss function, printing epoch progress (Loss/RMSE).
        5.  (Optional) Display a plot of the training RMSE over epochs.
        6.  Generate and print top-K recommendations for a sample `target_user_id` specified within the script's `if __name__ == "__main__":` block.

2.  **Modify Target User:**
    *   To get recommendations for a different user, change the `target_user_id` variable near the end of the script before running it.


