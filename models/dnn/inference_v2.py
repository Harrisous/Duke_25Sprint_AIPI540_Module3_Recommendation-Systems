import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = CURRENT_DIR / ".." / ".." / "data"

sys.path.append(CURRENT_DIR.absolute())

from model_v2 import HybridRecModel, user_ids, item_ids, user_llm_emb, item_llm_emb, user_id_map, item_id_map, item_llm_data


# Initialize model
model = HybridRecModel(
    num_users=len(user_ids),
    num_items=len(item_ids),
    id_emb_dim=64,
    llm_emb_dim=user_llm_emb.shape[1],
    user_llm_emb=user_llm_emb,
    item_llm_emb=item_llm_emb
)

model.load_state_dict(torch.load(CURRENT_DIR / "models" / "model_v2.pth"))

model.eval()

def get_recommendations(user_id, user_text=None, num_recommendations=10):
    uidx = user_id_map.get(str(user_id), None)
    if uidx is None:
        print(f"== User {user_id} not found, using fallback user_text: {user_text} ==")
        user_vec = model.get_fallback_user_with_text(user_text)
    else:
        user_vec = model.user_llm_emb[uidx].unsqueeze(0)
    
    recommendations = []
    for iidx in item_id_map.keys():
        pred = predict(model, user_vec, iidx)
        item_idx = item_id_map[iidx]  # Get movie position index
        recommendations.append((iidx, pred, item_llm_data.iloc[item_idx]["llm_text"]))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:num_recommendations]


def predict(model, user_vec, item_id):
    iidx = item_id_map.get(str(item_id), None)

    with torch.no_grad():
        if iidx is not None:
            item_vec = model.item_llm_emb[iidx].unsqueeze(0)
        else:
            print(f"== Item {item_id} not found, using fallback item ==")
            item_vec = model.get_fallback_item()

        # Process through MLP layers
        user_vec = model.user_mlp(user_vec)
        item_vec = model.item_mlp(item_vec)
        
        # 计算相似度并映射到1-5范围
        similarity = F.cosine_similarity(user_vec, item_vec)
        rating = 1 + 2 * (similarity + 1)  # 将[-1,1]映射到[1,5]

        return rating.item()

if __name__ == "__main__":
    # Example
    print("\nPredicting recommended movies for user:")
    for iidx, pred, text in get_recommendations(user_id=1, user_text="Age: 33, gender: M, occupation: none. Rating history includes: 'Star Trek III: The Search for Spock (1984)' (Genre: Action, Adventure, Sci-Fi) rated 5 stars", num_recommendations=10):
        print(f"Movie ID: {iidx}, Rating: {pred}, Description: {text}")
