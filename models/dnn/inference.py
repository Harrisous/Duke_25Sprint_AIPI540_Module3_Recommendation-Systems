import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = CURRENT_DIR / ".." / ".." / "data"

sys.path.append(CURRENT_DIR.absolute())

from model import (HybridRecModel, item_id_map, item_ids, item_llm_data,
                   item_llm_emb, user_id_map, user_ids, user_llm_emb)

# Initialize model
model = HybridRecModel(
    num_users=len(user_ids),
    num_items=len(item_ids),
    id_emb_dim=64,
    llm_emb_dim=user_llm_emb.shape[1],
    user_llm_emb=user_llm_emb,
    item_llm_emb=item_llm_emb,
)

model.load_state_dict(torch.load(CURRENT_DIR / "models" / "model.pth"))

model.eval()


def get_recommendations(user_id, user_text=None, num_recommendations=10):
    uidx = user_id_map.get(user_id, None)
    if uidx is None:
        user_vec = model.get_fallback_user_with_text(user_text)
    else:
        user_vec = torch.cat(
            [
                model.user_id_emb(torch.tensor([uidx])),
                model.user_llm_emb[uidx].unsqueeze(0),
            ],
            dim=1,
        )

    recommendations = []
    for iidx in item_id_map.keys():
        pred = predict(model, user_vec, iidx)
        item_idx = item_id_map[iidx]  # Get movie position index
        recommendations.append((iidx, pred, item_llm_data.iloc[item_idx]["llm_text"]))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:num_recommendations]


def predict(model, user_vec, item_id):
    iidx = item_id_map.get(item_id, None)

    with torch.no_grad():
        if iidx is not None:
            item_vec = torch.cat(
                [
                    model.item_id_emb(torch.tensor([iidx])),
                    model.item_llm_emb[iidx].unsqueeze(0),
                ],
                dim=1,
            )
        else:
            item_vec = model.get_fallback_item()

        # Process through MLP layers
        user_vec = model.user_mlp(user_vec)
        item_vec = model.item_mlp(item_vec)

        return (user_vec * item_vec).sum().item()


# Example
print("\nPredicting recommended movies for user:")
for iidx, pred, text in get_recommendations(
    user_id=9999,
    user_text="age: 24, gender: M, occupation: technician. I like Sci-Fi movies.",
    num_recommendations=10,
):
    print(f"Movie ID: {iidx}, Rating: {pred}, Description: {text}")
