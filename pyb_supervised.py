# https://aiml4os.github.io/funathon-project2/1-ttc.html

# %% Imports
# import mlflow
import polars as pl
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers.value_encoder import ValueEncoder
from torchTextClassifiers.tokenizers import WordPieceTokenizer

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

# %%
