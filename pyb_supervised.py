# %% Imports
import numpy as np
import mlflow
import polars as pl
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

load_dotenv(override=True)
# os.getenv('MLFLOW_TRACKING_USERNAME')

# %%
# Load dataset

df = pl.read_parquet("https://minio.lab.sspcloud.fr/" +
                     "projet-formation/diffusion/funathon/2026/project2/" +
                     "generation_None_temp08.parquet")
type(df)

print(f"Total rows: {len(df)}")
df.head(7)
df.tail(7)

n_classes = df['code'].n_unique()
print(f"Number of unique NACE codes: {n_classes}")

# %% SPLIT
train_df, tmp_df = train_test_split(df, test_size=0.3, random_state=954)
val_df, test_df  = train_test_split(tmp_df, test_size=0.5, random_state=954)

X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")

# %%
