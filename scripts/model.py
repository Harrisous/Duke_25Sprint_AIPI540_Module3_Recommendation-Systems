"""
Script to train and evaluate the recommendation model
"""
import pathlib
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

# Define paths
CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR = CURRENT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models" / "dnn"

# Add models directory to path for imports
sys.path.append(str(MODELS_DIR))

class RecommendationModel:
    """
    Class to handle model training and evaluation
    """
    def __init__(self):
        """Initialize the recommendation model"""
        self.load_data()
        self.setup_model()
    
    def load_data(self):
        """Load necessary data for model training"""
        print("Loading data...")
        
        # Load ratings data
        ratings_file = DATA_DIR / "raw" / "ml-100k" / "u.data"
        self.ratings = pd.read_csv(
            ratings_file, 
            sep="\t", 
            names=["user_id", "item_id", "rating", "timestamp"]
        )
        
        # Load ID mappings
        with open(DATA_DIR / "processed" / "user_id_map.json", "r") as f:
            self.user_id_map = json.load(f)
        
        with open(DATA_DIR / "processed" / "item_id_map.json", "r") as f:
            self.item_id_map = json.load(f)
        
        # Load embeddings
        user_llm_data = pd.read_json(DATA_DIR / "processed" / "user_embeddings.json", orient="records")
        item_llm_data = pd.read_json(DATA_DIR / "processed" / "movie_embeddings.json", orient="records")
        
        self.user_llm_emb = torch.tensor(user_llm_data["embedding"].tolist(), dtype=torch.float32)
        self.item_llm_emb = torch.tensor(item_llm_data["embedding"].tolist(), dtype=torch.float32)
        
        self.user_ids = self.ratings["user_id"].unique()
        self.item_ids = self.ratings["item_id"].unique()
        
        print(f"Loaded data for {len(self.user_ids)} users and {len(self.item_ids)} items.")
    
    def setup_model(self):
        """Set up the model architecture"""
        from model_v2 import HybridRecModel
        
        print("Setting up model...")
        
        self.model = HybridRecModel(
            num_users=len(self.user_ids),
            num_items=len(self.item_ids),
            id_emb_dim=64,
            llm_emb_dim=self.user_llm_emb.shape[1],
            user_llm_emb=self.user_llm_emb,
            item_llm_emb=self.item_llm_emb
        )
    
    def load_fold_data(self, fold_id):
        """Load training and test data for a specific fold"""
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
        
        # Map IDs to indices
        train_data["user_idx"] = train_data["user_id"].map(lambda x: self.user_id_map[str(x)])
        train_data["item_idx"] = train_data["item_id"].map(lambda x: self.item_id_map[str(x)])
        test_data["user_idx"] = test_data["user_id"].map(lambda x: self.user_id_map[str(x)])
        test_data["item_idx"] = test_data["item_id"].map(lambda x: self.item_id_map[str(x)])
        
        return train_data, test_data
    
    def evaluate_model(self, test_users, test_items, test_ratings, batch_size=1024):
        """Evaluate model performance"""
        self.model.eval()
        total_mse = 0
        total_mae = 0
        n_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(test_users), batch_size):
                batch_users = test_users[i:i+batch_size]
                batch_items = test_items[i:i+batch_size]
                batch_ratings = test_ratings[i:i+batch_size]
                
                preds = self.model(batch_users, batch_items)
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
    
    def train_fold(self, fold_id, epochs=10, batch_size=1024, lr=0.005):
        """Train the model on a specific fold"""
        print(f"\nTraining on fold {fold_id}...")
        
        # Load data for this fold
        train_df, test_df = self.load_fold_data(fold_id)
        
        # Prepare training data
        train_users = torch.LongTensor(train_df["user_idx"].values)
        train_items = torch.LongTensor(train_df["item_idx"].values)
        train_ratings = torch.FloatTensor(train_df["rating"].values)
        
        # Prepare test data
        test_users = torch.LongTensor(test_df["user_idx"].values)
        test_items = torch.LongTensor(test_df["item_idx"].values)
        test_ratings = torch.FloatTensor(test_df["rating"].values)
        
        # Reinitialize model
        self.model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        # Training process
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(train_users), batch_size):
                batch_users = train_users[i:i+batch_size]
                batch_items = train_items[i:i+batch_size]
                batch_ratings = train_ratings[i:i+batch_size]
                
                optimizer.zero_grad()
                preds = self.model(batch_users, batch_items)
                
                # Basic MSE loss
                loss = criterion(preds, batch_ratings)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        
        # Evaluate
        metrics = self.evaluate_model(test_users, test_items, test_ratings)
        print("Evaluation results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        return metrics
    
    def cross_validation(self, n_folds=5, epochs=20):
        """Perform n-fold cross validation"""
        print(f"Starting {n_folds}-fold cross validation...")
        all_metrics = []
        
        for fold_id in range(1, n_folds + 1):
            metrics = self.train_fold(fold_id, epochs=epochs)
            all_metrics.append(metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        std_metrics = {}
        for metric in all_metrics[0].keys():
            values = [m[metric] for m in all_metrics]
            avg_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)
            print(f"\n{metric} average: {avg_metrics[metric]:.4f} (±{std_metrics[metric]:.4f})")
        
        return avg_metrics, std_metrics
    
    def train_final_model(self, epochs=30):
        """Train final model on all data"""
        print("\nTraining final model on all data...")
        
        # Prepare all training data
        self.ratings["user_idx"] = self.ratings["user_id"].map(lambda x: self.user_id_map[str(x)])
        self.ratings["item_idx"] = self.ratings["item_id"].map(lambda x: self.item_id_map[str(x)])
        
        train_users = torch.LongTensor(self.ratings["user_idx"].values)
        train_items = torch.LongTensor(self.ratings["item_idx"].values)
        train_ratings = torch.FloatTensor(self.ratings["rating"].values)
        
        # Reinitialize model
        self.model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        optimizer = optim.Adam(self.model.parameters(), lr=0.005, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        # Training process
        self.model.train()
        batch_size = 1024
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(train_users), batch_size):
                batch_users = train_users[i:i+batch_size]
                batch_items = train_items[i:i+batch_size]
                batch_ratings = train_ratings[i:i+batch_size]
                
                optimizer.zero_grad()
                preds = self.model(batch_users, batch_items)
                loss = criterion(preds, batch_ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        
        # Save model
        model_dir = MODELS_DIR / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_dir / "model_v2.pth")
        print(f"Model saved to {model_dir / 'model_v2.pth'}")

def main():
    """Main function to train and evaluate model"""
    print("Training recommendation model...")
    
    # Initialize model
    rec_model = RecommendationModel()
    
    # Perform cross-validation
    avg_metrics, std_metrics = rec_model.cross_validation()
    
    # Train final model on all data
    rec_model.train_final_model()
    
    print("\nModel training completed.")
    print("\nNext steps:")
    print("1. Start the API server: python models/dnn/api_server.py")

if __name__ == "__main__":
    main()
