import json
import os
import pathlib

import pandas as pd

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = CURRENT_DIR / ".." / ".." / "data"

# Create ID mapping tables (using all data)
ratings = pd.read_csv(
    DATA_DIR / "raw" / "ml-100k" / "u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"],
)
user_ids = sorted(ratings["user_id"].unique())
item_ids = sorted(ratings["item_id"].unique())
user_id_map = {str(uid): idx for idx, uid in enumerate(user_ids)}
item_id_map = {str(iid): idx for idx, iid in enumerate(item_ids)}

# Save ID mapping tables
with open(DATA_DIR / "processed" / "user_id_map.json", "w") as f:
    json.dump(user_id_map, f, indent=4)
with open(DATA_DIR / "processed" / "item_id_map.json", "w") as f:
    json.dump(item_id_map, f, indent=4)
