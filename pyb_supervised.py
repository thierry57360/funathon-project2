# https://aiml4os.github.io/funathon-project2/1-ttc.html

# %% Imports
import mlflow
import polars as pl
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers.value_encoder import ValueEncoder
from torchTextClassifiers.tokenizers import WordPieceTokenizer
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
import s3fs
import random

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
val_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=954)

X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")

# %% code to integers
all_codes = set(df['code'])
train_codes = set(train_df['code'])
missing = all_codes - train_codes

if missing:
    print(f"WARNING: {len(missing)} code(s) missing from training set: {missing}")
else:
    print(f"OK — all {len(all_codes)} codes appear in the training set.")

encoder = LabelEncoder()
encoder.fit(train_df['code'].to_numpy())

# to get back code instead of integers
value_encoder = ValueEncoder(label_encoder=encoder)

# %% Train the tokenizer

tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=10)
tokenizer.train(X_train)

# (A Tensor is a multi-dimensional matrix containing elements of a single data type)
print("Output tensor size:", tokenizer.tokenize(X_train[0]).input_ids.shape)
print("Vocabulary size:", tokenizer.vocab_size)

# %%
# Look at an example of tokenization
print("Raw text : ", X_train[554])
print(
    "Tokens id:",
    tokenizer.tokenize(X_train[554]).input_ids.squeeze(0)
)
print(
    "Tokens:",
    tokenizer.tokenizer.convert_ids_to_tokens(
        tokenizer.tokenize(X_train[554]).input_ids.squeeze(0)
    )
)

# %% create the classifier
model_config = ModelConfig(
    embedding_dim=96,
    num_classes=n_classes
)

ttc = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    value_encoder=value_encoder,
)

# %%  Prepare training
training_config = TrainingConfig(
    lr=5e-4,  # a standard starting learning rate for Adam-based optimizers
    batch_size=128,  # examples processed before wt update, larger are faster but more memory
    num_epochs=1,  # one pass over the data, kept short for this demo
    patience_early_stopping=5  # to avoid overfitting
)

# %% Train on a small subsample
mlflow.set_experiment("funathon-2026-project2_PYB")
mlflow.pytorch.autolog()

with mlflow.start_run() as run:
    # This should take approximately 1-2 min
    ttc.train(
        X_train,
        y_train,
        training_config=training_config,
        X_val=X_val,
        y_val=y_val,
        verbose=True,
    )

    mlflow.log_artifacts(
        training_config.save_path,   # local folder produced by ttc.train()
        artifact_path="model_artifacts",
    )

# # %%
# #| label: load-from-run
# #| code-overflow: scroll
# #| output: true
# local_dir = mlflow.artifacts.download_artifacts(
#     f"runs:/{run.info.run_id}/model_artifacts"
# )

# # Rebuild the torchTextClassifiers object from the downloaded files
# ttc_loaded = torchTextClassifiers.load(local_dir)

# print(ttc_loaded)

# %% Load a better model

fs = s3fs.S3FileSystem(
    anon=True,  # public bucket
    endpoint_url="https://minio.lab.sspcloud.fr",
)

local_dir = "./mlflow-artifacts/"
fs.get(
    "projet-funathon/diffusion/mlflow-artifacts/",
    local_dir,
    recursive=True,
)
# Rebuild the torchTextClassifiers object from the downloaded files
ttc = torchTextClassifiers.load(local_dir)

ttc.pytorch_model.eval()

# %% Generate top-5 predictions with confidence scores