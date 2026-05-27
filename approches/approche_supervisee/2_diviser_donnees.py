# %%
import polars as pl
df = pl.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
)
print(df.head())
print(f"total rows: {len(df)}")

# %%
from sklearn.model_selection import train_test_split
# importe la fonction qui permet de découper les données en plusieurs jeux
# %%
train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
# découpe les données en un jeu train de 70 % et un jeu temporaire de 30 %

val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=42)
# découpe le jeu temporaire en deux parties égales pour obtenir validation et test

X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
# crée les données d'entrée et les codes à prédire pour le jeu train

X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
# crée les données d'entrée et les codes à prédire pour le jeu validation

X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()
# crée les données d'entrée et les codes à prédire pour le jeu test

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
# affiche le nombre de lignes dans chaque jeu de données

# Le modèle va apprendre sur 49K exemples (70%)
# on va le surveiller avec 10,5KL exemples de validation (15%)
# # puis on le testera à la fin sur 10 500 exemples mis de côté (15%)

# %%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
# crée un encodeur pour convertir les codes nace en nombres

encoder.fit(train_df["code"].to_numpy())
# apprend la liste des codes nace présents dans la var code du jeu train

# %%
all_codes = set(df["code"])
# récupère tous les codes nace présents dans la var code du jeu complet

train_codes = set(train_df["code"])
# récupère tous les codes nace présents dans la var code du jeu train

missing = all_codes - train_codes
# identifie les codes nace présents dans le jeu complet mais absents du jeu train

if missing:
    # vérifie s'il existe au moins un code nace absent du jeu train

    print(f"warning: {len(missing)} code(s) missing from training set: {missing}")
    # affiche un avertissement avec le nombre de codes absents et leur liste

else:
    # cas où aucun code nace ne manque dans le jeu train

    print(f"ok — all {len(all_codes)} codes appear in the training set")
    # confirme que tous les codes nace sont bien présents dans le jeu train

# %%
from torchTextClassifiers.value_encoder import ValueEncoder
# importe l'outil valueencoder utilisé par torchtextclassifiers pour gérer les codes à prédire

value_encoder = ValueEncoder(label_encoder=encoder)
# crée un valueencoder à partir de l'encodeur sklearn déjà entraîné sur les codes nace du jeu train
# %%
