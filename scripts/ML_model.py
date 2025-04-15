import numpy as np
import pandas as pd
import os
import pathlib
import requests # For downloading data if needed
import zipfile # For downloading data if needed
import io # For downloading data if needed
import shutil # For removing temp download dir

# --- PyTorch Specific Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Configuration ---
CURRENT_DIR = pathlib.Path().resolve()
# Adjust if your structure differs
# Use '.' if data is in the root project folder
DATA_DIR = CURRENT_DIR / "data" if (CURRENT_DIR / "data").exists() else CURRENT_DIR
RAW_DATA_DIR = DATA_DIR / "raw"
MOVIELENS_DIR = RAW_DATA_DIR / "ml-100k"
PROCESSED_DATA_DIR = DATA_DIR / "processed" # Not used by AutoRec but kept for consistency


# --- 1. Load and Prepare Data ---
print("Loading and preparing data for AutoRec (PyTorch)...")
ratings_file = MOVIELENS_DIR / "u.data"
movies_file_path = MOVIELENS_DIR / "u.item"

try:
    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(ratings_file, sep='\t', names=ratings_cols, engine='python')
    print(f"Loaded {len(ratings_df)} ratings.")

    max_user_id = ratings_df['user_id'].max()
    max_item_id = ratings_df['item_id'].max()
    print(f"Max User ID: {max_user_id}, Max Item ID: {max_item_id}")

    # Create the User-Item Rating Matrix (Users as rows, Items as columns)
    rating_matrix = ratings_df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
    # Reindex to ensure matrix covers all IDs from 1 to max
    rating_matrix = rating_matrix.reindex(index=range(1, max_user_id + 1), columns=range(1, max_item_id + 1), fill_value=0)

    rating_matrix_np = rating_matrix.values.astype(np.float32) # Use float32

    actual_num_users = rating_matrix_np.shape[0]
    actual_num_items = rating_matrix_np.shape[1]
    print(f"Rating matrix shape: {rating_matrix_np.shape}") # (Users x Items)

    # Convert to PyTorch Tensor
    rating_tensor = torch.FloatTensor(rating_matrix_np)

    # Create Dataset and DataLoader (each sample is a user's rating vector)
    dataset = TensorDataset(rating_tensor) # Input and target are the same row

except FileNotFoundError:
     print(f"Error: Ratings file not found at {ratings_file}")
     exit()
except Exception as e:
    print(f"An error occurred during data preparation: {e}")
    exit()


# --- 2. Define PyTorch AutoRec Custom Loss Function ---
# Loss should only be calculated for ratings that actually exist (non-zero)
def masked_mse_loss(y_true, y_pred):
    """
    Calculates Mean Squared Error only on non-zero entries in y_true (PyTorch version).
    """
    mask = (y_true != 0).float() # Create a mask for observed ratings
    squared_error = (y_pred - y_true).pow(2)
    masked_squared_error = squared_error * mask
    # Add small epsilon to avoid division by zero
    loss = torch.sum(masked_squared_error) / (torch.sum(mask) + 1e-8)
    return loss

# --- 3. Build the Shallow Autoencoder Model (U-AutoRec) in PyTorch ---
class AutoRec(nn.Module):
    def __init__(self, num_items, hidden_units=512):
        super(AutoRec, self).__init__()
        # Encoder/Hidden Layer
        self.encoder = nn.Linear(num_items, hidden_units)
        self.activation = nn.ReLU() # Or nn.Sigmoid(), nn.Tanh()
        # Optional: Dropout
        # self.dropout = nn.Dropout(0.5)
        # Decoder/Output Layer
        self.decoder = nn.Linear(hidden_units, num_items)

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        # x = self.dropout(x) # Apply dropout if used
        x = self.decoder(x)
        return x

# Model Hyperparameters
HIDDEN_UNITS = 256
L2_REG = 0.001 # Corresponds to weight_decay in Adam optimizer
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 128

print("\nBuilding AutoRec model (PyTorch)...")
autorec_model = AutoRec(num_items=actual_num_items, hidden_units=HIDDEN_UNITS)

# --- Setup Device (GPU if available, else CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
autorec_model.to(device)

# --- Define Optimizer ---
# Use weight_decay for L2 regularization
optimizer = optim.Adam(autorec_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)

# --- Create DataLoader ---
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 4. Train the AutoRec Model (PyTorch Training Loop) ---
print("\nStarting AutoRec model training (PyTorch)...")
train_losses = []
for epoch in range(EPOCHS):
    autorec_model.train() # Set model to training mode
    epoch_loss = 0.0
    num_batches = 0

    for batch_data in dataloader:
        # batch_data is a list containing one tensor (the user vectors)
        user_vectors = batch_data[0].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        reconstructed_vectors = autorec_model(user_vectors)

        # Calculate loss (only on observed ratings)
        loss = masked_mse_loss(user_vectors, reconstructed_vectors) # Input is also the target

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_epoch_loss = epoch_loss / num_batches
    epoch_rmse = np.sqrt(avg_epoch_loss) # Calculate RMSE from average MSE loss
    train_losses.append(epoch_rmse)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss (MSE): {avg_epoch_loss:.6f}, RMSE: {epoch_rmse:.6f}")

print("Training complete.")

# --- 5. Make Recommendations using the Trained Model (PyTorch) ---

# Load movie titles
try:
    movie_cols = ['item_id', 'title']
    movies_df = pd.read_csv(
        movies_file_path, sep='|', encoding='latin-1', header=None,
        usecols=[0, 1], names=movie_cols, index_col='item_id' # Use item_id as index
    )
    titles_loaded = True
    print("\nLoaded movie titles.")
except FileNotFoundError:
    print(f"Warning: Movie titles file not found at {movies_file_path}. Cannot display titles.")
    titles_loaded = False

def recommend_autorec_pytorch(user_id, model, full_rating_matrix_np, movies_info_df, top_k=10, device='cpu'):
    """
    Generates recommendations for a given user_id using the trained PyTorch AutoRec model.
    """
    model.eval() # Set model to evaluation mode

    if user_id < 1 or user_id > full_rating_matrix_np.shape[0]:
        print(f"Error: User ID {user_id} is out of valid range (1-{full_rating_matrix_np.shape[0]})")
        return []

    user_index = user_id - 1

    # Get the user's actual rating vector and convert to tensor
    user_vector_np = full_rating_matrix_np[user_index].reshape(1, -1)
    user_vector_tensor = torch.FloatTensor(user_vector_np).to(device)

    # Predict ratings (no gradients needed)
    with torch.no_grad():
        predicted_ratings_vector = model(user_vector_tensor)[0] # Get the single output vector

    # Move predictions back to CPU and convert to numpy
    predicted_ratings_np = predicted_ratings_vector.cpu().numpy()
    original_ratings_np = user_vector_np[0] # Original ratings for filtering

    # Get indices of items the user has NOT rated (original rating was 0)
    unrated_indices = np.where(original_ratings_np == 0)[0]

    if len(unrated_indices) == 0:
        print(f"User {user_id} has rated all items. Cannot recommend.")
        return []

    # Get predicted ratings ONLY for unrated items
    predicted_unrated_scores = predicted_ratings_np[unrated_indices]

    # Sort the unrated items by predicted score (descending)
    # Need to handle potential NaN predictions if activation/loss produced them (unlikely with linear output)
    valid_predictions_mask = ~np.isnan(predicted_unrated_scores)
    indices_to_sort = unrated_indices[valid_predictions_mask]
    scores_to_sort = predicted_unrated_scores[valid_predictions_mask]

    if len(scores_to_sort) == 0:
        print(f"No valid predictions for unrated items for user {user_id}.")
        return []

    sorted_indices_local = np.argsort(scores_to_sort)[::-1]

    # Get the original indices (in the full item range) of the top_k recommendations
    top_k_item_indices = indices_to_sort[sorted_indices_local[:top_k]]

    # Item indices are 0-based, map back to original item IDs (which are index + 1)
    recommended_item_ids = top_k_item_indices + 1

    # Prepare output with titles and scores
    recommendations = []
    if titles_loaded:
        for i, item_id in enumerate(recommended_item_ids):
            # Score corresponds to the sorted prediction
            score = scores_to_sort[sorted_indices_local[i]]
            try:
                title = movies_info_df.loc[item_id]['title']
                 # Skip 'unknown' titles
                if title.strip().lower() == 'unknown':
                    print(f"Skipping item ID {item_id} with 'unknown' title.")
                    continue
                recommendations.append({'item_id': item_id, 'title': title, 'predicted_rating': score})
                if len(recommendations) >= top_k: # Ensure we don't exceed top_k after skipping
                    break
            except KeyError:
                print(f"Warning: Could not find title for item ID {item_id}. Skipping.")
                continue
    else:
         for i, item_id in enumerate(recommended_item_ids):
             score = scores_to_sort[sorted_indices_local[i]]
             recommendations.append({'item_id': item_id, 'predicted_rating': score})
             if len(recommendations) >= top_k:
                 break

    return recommendations

# --- Example Usage ---
if __name__ == "__main__":
    target_user_id = 50 # Example user ID (1 to 943)
    print(f"\n--- Generating recommendations for User {target_user_id} using AutoRec (PyTorch) ---")

    # Pass the numpy matrix, not the tensor, to the recommendation function
    recommendations = recommend_autorec_pytorch(target_user_id, autorec_model, rating_matrix_np, movies_df, top_k=10, device=device)

    if recommendations:
        print("\n--- Top Recommendations ---")
        for i, rec in enumerate(recommendations):
             title_str = f"{rec['title']} (ID: {rec['item_id']})" if titles_loaded else f"Item ID: {rec['item_id']}"
             print(f"{i+1}. {title_str} - Predicted Rating: {rec['predicted_rating']:.4f}")
    else:
        print(f"\nNo recommendations could be generated for user {target_user_id}.")
    
    # --- After the training loop ---
    #print("Saving trained model state...")
    MODEL_SAVE_PATH = 'autorec_model_state.pth' # Save in the same directory as the script for simplicity
    torch.save(autorec_model.state_dict(), MODEL_SAVE_PATH)
    #print(f"Model state saved to {MODEL_SAVE_PATH}")


    # Optional: Plot training RMSE
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_losses, marker='o', linestyle='-')
    plt.title('Training RMSE per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.show()
