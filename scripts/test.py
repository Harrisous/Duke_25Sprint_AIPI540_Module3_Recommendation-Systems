import pandas as pd

# Load u.item
movies_df = pd.read_csv(
    "data/raw/ml-100k/u.item", 
    sep='|', 
    encoding='latin-1', 
    header=None, 
    usecols=[0, 1], 
    names=['item_id', 'title']
)

# Check if ID 267 exists
print(movies_df[movies_df['item_id'] == 267]['title'])
