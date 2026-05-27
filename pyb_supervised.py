# %%
import mlflow
import polars as pl
from dotenv import load_dotenv

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

# %% Count unique NACE codes
n_classes = df['code'].n_unique()
print(f"Number of unique NACE codes: {n_classes}")

# %%
