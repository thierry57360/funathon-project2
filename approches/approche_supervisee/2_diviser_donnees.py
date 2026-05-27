# %%
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pl.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
)
# charge le fichier parquet contenant les textes d'activité et les codes nace associés

print(df.head())
# affiche les premières lignes pour vérifier le chargement

print(f"total rows: {len(df)}")
# affiche le nombre total de lignes du jeu complet

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

print(f"train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
# affiche le nombre de lignes dans chaque jeu de données

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
    print(f"warning: {len(missing)} code(s) missing from training set: {missing}")
    # affiche un avertissement avec le nombre de codes absents et leur liste

else:
    print(f"ok — all {len(all_codes)} codes appear in the training set")
    # confirme que tous les codes nace sont bien présents dans le jeu train
    
# %%
import sys

venv_site_packages = "/home/onyxia/work/funathon-project2/.venv/lib/python3.13/site-packages"

if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

print("venv ajouté au path")


# %%
import torchTextClassifiers
print("torchtextclassifiers ok")


# %%
from torchTextClassifiers.value_encoder import ValueEncoder
# importe l'outil valueencoder utilisé par torchtextclassifiers pour gérer les codes à prédire

value_encoder = ValueEncoder(label_encoder=encoder)
# crée un valueencoder à partir de l'encodeur sklearn déjà entraîné sur les codes nace du jeu train

# %%
