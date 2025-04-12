import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = CURRENT_DIR / ".." / ".." / "data"

sys.path.append(CURRENT_DIR.absolute())

from get_profile_embedding import get_profile_embedding

def load_fold_data(fold_id):
    """Load training and test data for the specified fold"""
    train_data = pd.read_csv(
        DATA_DIR / "raw" / "ml-100k" / f"u{fold_id}.base",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    test_data = pd.read_csv(
        DATA_DIR / "raw" / "ml-100k" / f"u{fold_id}.test",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    return train_data, test_data

# Create ID mapping tables (using all data)
ratings = pd.read_csv(DATA_DIR / "raw" / "ml-100k" / "u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
user_ids = ratings["user_id"].unique()
item_ids = ratings["item_id"].unique()
user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}

# Load cached LLM embeddings
user_llm_data = pd.read_json(DATA_DIR / "processed" / "user_embeddings.json", orient="records")
item_llm_data = pd.read_json(DATA_DIR / "processed" / "movie_embeddings.json", orient="records")

user_llm_emb = torch.tensor(user_llm_data["embedding"].tolist(), dtype=torch.float32)
item_llm_emb = torch.tensor(item_llm_data["embedding"].tolist(), dtype=torch.float32)

# Model definition
class HybridRecModel(nn.Module):
    def __init__(self, num_users, num_items, id_emb_dim, llm_emb_dim, user_llm_emb, item_llm_emb):
        super().__init__()

        self.user_llm_emb = user_llm_emb  # 固定不训练
        self.item_llm_emb = item_llm_emb

        self.total_dim = llm_emb_dim

        self.user_mlp = nn.Sequential(
            nn.Linear(self.total_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(self.total_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )


    def forward(self, user_idx, item_idx):
        ullm_emb = self.user_llm_emb[user_idx]
        illm_emb = self.item_llm_emb[item_idx]

        user_vec = ullm_emb
        item_vec = illm_emb

        user_vec = self.user_mlp(user_vec)
        item_vec = self.item_mlp(item_vec)

        return (user_vec * item_vec).sum(dim=1)

    def get_fallback_user(self):
        return self.user_llm_emb.mean(dim=0, keepdim=True)
    
    def get_fallback_user_with_text(self, user_text):
        user_llm_emb = get_profile_embedding(user_text)
        user_llm_emb = torch.tensor(user_llm_emb, dtype=torch.float32).unsqueeze(0)
        return user_llm_emb

    def get_fallback_item(self):
        return self.item_llm_emb.mean(dim=0, keepdim=True)

def evaluate_model(model, test_users, test_items, test_ratings, batch_size=1024):
    """Evaluate model performance"""
    model.eval()
    total_mse = 0
    total_mae = 0
    n_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(test_users), batch_size):
            batch_users = test_users[i:i+batch_size]
            batch_items = test_items[i:i+batch_size]
            batch_ratings = test_ratings[i:i+batch_size]
            
            preds = model(batch_users, batch_items)
            mse = ((preds - batch_ratings) ** 2).mean().item()
            mae = (preds - batch_ratings).abs().mean().item()
            
            total_mse += mse
            total_mae += mae
            n_batches += 1
    
    avg_mse = total_mse / n_batches
    avg_mae = total_mae / n_batches
    rmse = np.sqrt(avg_mse)
    
    return {
        'MSE': avg_mse,
        'RMSE': rmse,
        'MAE': avg_mae
    }

def train_and_evaluate_fold(fold_id, model, epochs=10, batch_size=1024, lr=0.005):
    """Train and evaluate a single fold"""
    print(f"\nStarting training for Fold {fold_id}")
    
    # Load data for this fold
    train_df, test_df = load_fold_data(fold_id)
    train_df["user_idx"] = train_df["user_id"].map(user_id_map)
    train_df["item_idx"] = train_df["item_id"].map(item_id_map)
    test_df["user_idx"] = test_df["user_id"].map(user_id_map)
    test_df["item_idx"] = test_df["item_id"].map(item_id_map)
    
    # Prepare training data
    train_users = torch.LongTensor(train_df["user_idx"].values)
    train_items = torch.LongTensor(train_df["item_idx"].values)
    train_ratings = torch.FloatTensor(train_df["rating"].values)
    
    # Prepare test data
    test_users = torch.LongTensor(test_df["user_idx"].values)
    test_items = torch.LongTensor(test_df["item_idx"].values)
    test_ratings = torch.FloatTensor(test_df["rating"].values)
    
    # Reinitialize model
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training process
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(train_users), batch_size):
            batch_users = train_users[i:i+batch_size]
            batch_items = train_items[i:i+batch_size]
            batch_ratings = train_ratings[i:i+batch_size]
            
            optimizer.zero_grad()
            preds = model(batch_users, batch_items)
            loss = criterion(preds, batch_ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    # Evaluation
    metrics = evaluate_model(model, test_users, test_items, test_ratings)
    return metrics

def run_cross_validation(model, n_folds=5):
    """Perform n-fold cross validation"""
    all_metrics = []
    
    for fold_id in range(1, n_folds + 1):
        metrics = train_and_evaluate_fold(fold_id, model, epochs=20)
        all_metrics.append(metrics)
        print(f"\nFold {fold_id} evaluation results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    # Calculate average metrics
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        values = [m[metric] for m in all_metrics]
        avg_metrics[metric] = np.mean(values)
        std_metrics = np.std(values)
        print(f"\n{metric} average: {avg_metrics[metric]:.4f} (±{std_metrics:.4f})")
    
    return avg_metrics



if __name__ == "__main__":
    # Initialize model
    model = HybridRecModel(
        num_users=len(user_ids),
        num_items=len(item_ids),
        id_emb_dim=64,
        llm_emb_dim=user_llm_emb.shape[1],
        user_llm_emb=user_llm_emb,
        item_llm_emb=item_llm_emb
    )

    # Perform 5-fold cross validation
    print("\nStarting 5-fold cross validation evaluation...")
    final_metrics = run_cross_validation(model)

    # save model
    MODEL_PATH = CURRENT_DIR / "models"
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH / "model_v2.pth")
