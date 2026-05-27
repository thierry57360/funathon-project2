# %%
import mlflow
# Importe la bibliothèque MLflow.
# MLflow sert à suivre l'xp de ML : paramètres, métriques, modèles sauvegardés, artefacts, etc

from dotenv import load_dotenv
# Importe la fonction load_dotenv, qui sert à charger les variables stockées dans le fichier .env.

load_dotenv(override=True)
# Charge les variables d'environnement depuis le fichier .env.
# override=True signifie que les valeurs du .env remplacent les valeurs déjà existantes
# Ici, cela permet de récupérer :
# - MLFLOW_TRACKING_URI
# - MLFLOW_TRACKING_USERNAME
# - MLFLOW_TRACKING_PASSWORD

# %%
# Nouvelle cellule indépendante.

import polars as pl
# Importe la bibliothèque Polars sous l'alias pl
# Polars sert à manipuler des tableaux de données, un peu comme pandas

df = pl.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
)
# Lit un fichier Parquet distant depuis S3
# Le fichier contient les données d'entraînement :
# - des descriptions textuelles d'activités
# - les codes NACE associés
# Le résultat est stocké dans la variable df, sous forme de DataFrame Polars

print(df.head())
# Affiche les premières lignes du tableau
# Cela permet de vérifier rapidement que les données ont bien été chargées
# et de voir à quoi ressemblent les colonnes

print(f"Total rows: {len(df)}")
# Affiche le nombre total de lignes du tableau
# len(df) donne le nombre d'observations dans le DataFrame
# Le f-string permet d'insérer directement ce nombre dans le texte affiché
# %%
n_classes = df["code"].n_unique()
# Compte le nombre de codes différents présents dans la var "code"
# Chaque code unique correspond à une classe que le modèle devra apprendre à prédire
# Le résultat est stocké dans la variable n_classes

print(f"Number of unique NACE codes: {n_classes}")
# Affiche le nombre total de codes NACE distincts
# Le f-string permet d'insérer directement la valeur de n_classes dans le texte affiché
# %%
from sklearn.preprocessing import LabelEncoder
# importe l'outil qui transforme des valeurs texte en identifiants numériques

encoder = LabelEncoder()
# crée un encodeur pour convertir les codes nace en nombres

encoder.fit(train_df["code"].to_numpy())
# apprend la liste des codes nace présents dans la var code du jeu train
# %%
