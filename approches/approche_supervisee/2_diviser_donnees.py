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
from torchTextClassifiers.tokenizers import WordPieceTokenizer
# importe le tokenizer wordpiece utilisé pour transformer les textes en identifiants numériques

tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=10)
# crée un tokenizer avec un vocabulaire maximum de 5000 tokens et une longueur de sortie fixée à 10

tokenizer.train(X_train)
# entraîne le tokenizer sur les textes du jeu train

print("Output tensor size:", tokenizer.tokenize(X_train[0]).input_ids.shape)
# affiche la taille du tenseur produit après tokenisation du premier texte du jeu train

print("Vocabulary size:", tokenizer.vocab_size)
# affiche la taille du vocabulaire appris par le tokenizer

print("Raw text", X_train[0])
# affiche le texte brut avant tokenisation

print(
    "Tokens id:",
    tokenizer.tokenize(X_train[0]).input_ids.squeeze(0)
)
# affiche les identifiants numériques des tokens obtenus pour le premier texte

print(
    "Tokens:",
    tokenizer.tokenizer.convert_ids_to_tokens(
        tokenizer.tokenize(X_train[0]).input_ids.squeeze(0)
    )
)
# reconvertit les identifiants numériques en tokens lisibles pour comprendre le découpage du texte

# %%
n_classes = df["code"].n_unique()
# compte le nombre de codes nace différents présents dans la var code

print(f"number of unique nace codes: {n_classes}")
# affiche le nombre de classes que le modèle devra prédire

# %%
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
# importe les outils nécessaires pour configurer le modèle et l'entraînement

embedding_dim = 96
# fixe la taille des vecteurs qui représenteront les tokens

model_config = ModelConfig(
    embedding_dim=embedding_dim,
    num_classes=n_classes,
)
# crée la configuration du modèle avec la taille des embeddings et le nombre de classes à prédire

ttc = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    value_encoder=value_encoder,
)
# crée le classifieur en combinant le tokenizer, la configuration du modèle et l'encodeur des codes nace
# %%
training_config = TrainingConfig(
    num_epochs=1,
    # fixe le nombre de passages complets sur le jeu train

    batch_size=128,
    # fixe le nombre d'exemples traités avant chaque mise à jour du modèle

    lr=5 * 1e-4,
    # fixe le learning rate utilisé pour ajuster les poids du modèle

    patience_early_stopping=5,
    # arrête l'entraînement si la perte de validation ne s'améliore plus pendant 5 epochs
)
# crée la configuration d'entraînement du modèle
# %%
import mlflow
from dotenv import load_dotenv

load_dotenv(override=True)
# charge les identifiants mlflow depuis le fichier .env
# %%
mlflow.set_experiment("funathon-2026-project2")
# définit le nom de l'expérience mlflow dans laquelle les résultats seront enregistrés

mlflow.pytorch.autolog()
# active l'enregistrement automatique des informations liées à pytorch dans mlflow

with mlflow.start_run() as run:
    # démarre une nouvelle exécution mlflow pour suivre cet entraînement

    # this should take approximately 1-2mn
    # indique que l'entraînement devrait prendre environ 1 à 2 minutes

    ttc.train(
        X_train,
        y_train,
        training_config=training_config,
        X_val=X_val,
        y_val=y_val,
        verbose=True,
    )
    # entraîne le modèle sur les textes et codes du jeu train
    # utilise le jeu validation pour suivre les performances pendant l'entraînement
    # verbose=true affiche les informations d'entraînement dans la sortie

    mlflow.log_artifacts(
        training_config.save_path,
        # indique le dossier local où ttc.train a sauvegardé les fichiers du modèle

        artifact_path="model_artifacts",
        # définit le dossier mlflow dans lequel les fichiers du modèle seront stockés
    )
    # enregistre dans mlflow les fichiers produits par l'entraînement

# %%
import s3fs

fs = s3fs.S3FileSystem(
    anon=True,
    endpoint_url="https://minio.lab.sspcloud.fr",
)
# crée une connexion anonyme au stockage minio public

local_dir = "./mlflow-artifacts/"
# définit le dossier local où seront téléchargés les artefacts du modèle

fs.get(
    "projet-funathon/diffusion/mlflow-artifacts/",
    local_dir,
    recursive=True,
)
# télécharge récursivement les artefacts du modèle pré-entraîné depuis minio

ttc = torchTextClassifiers.load(local_dir)
# recharge le modèle pré-entraîné à partir des fichiers téléchargés

ttc.pytorch_model.eval()
# passe le modèle en mode évaluation pour faire des prédictions

# patch avec modele local sauvegardé
# %%
#from torchTextClassifiers import torchTextClassifiers

#ttc = torchTextClassifiers.load("my_ttc")
#ttc.pytorch_model.eval()

# passe le modèle en mode évaluation
# désactive les comportements propres à l'entraînement comme le dropout
# prépare le modèle pour faire des prédictions
# %%
import random

random_indices = random.sample(range(len(X_test)), 3)
example_texts = X_test[random_indices]
example_true_codes = y_test[random_indices]
print(example_texts)
top_k = 5
results = ttc.predict(example_texts, top_k=top_k, explain_with_captum=True)
for i, text in enumerate(example_texts):
    predicted_codes = [results["prediction"][i][k] for k in range(top_k)]
    confidence = [results["confidence"][i][k].item() for k in range(top_k)]
    print(f"\nText: {text}")
    print(f"  True code: {example_true_codes[i]}")
    for code, conf in zip(predicted_codes, confidence):
        print(f"  {code}  (confidence: {conf:.3f})")
# %%
results_test = ttc.predict(X_test, top_k=1)
# génère une prédiction pour chaque texte du jeu test
# top_k=1 signifie qu'on garde seulement le code nace le plus probable

preds = results_test["prediction"].squeeze(1)
# récupère les codes nace prédits par le modèle
# squeeze(1) enlève une dimension inutile du tableau de prédictions

accuracy = (preds == y_test).mean()
# compare les prédictions aux vrais codes nace du jeu test
# calcule la part de prédictions correctes

print(f"Test accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")
# affiche l'accuracy sur le jeu test
# affiche aussi le nombre de prédictions correctes sur le nombre total d'exemples test

# %%
from torchTextClassifiers.utilities.plot_explainability import (
    map_attributions_to_char, map_attributions_to_word,
    plot_attributions_at_char, plot_attributions_at_word, figshow,
)
# importe les outils permettant de transformer et visualiser les scores d'explicabilité

text_idx = 0
# choisit le texte à analyser parmi les exemples prédits

top_k_idx = 0
# choisit la prédiction à expliquer parmi les top prédictions
# ici 0 signifie qu'on explique la prédiction la plus probable

text_sample = example_texts[text_idx]
# récupère le texte brut que l'on veut expliquer

offsets = results["offset_mapping"][text_idx]
# récupère la position de chaque token dans le texte original

word_ids = results["word_ids"][text_idx]
# récupère l'identifiant du mot associé à chaque token

predicted_code = results["prediction"][text_idx][top_k_idx]
# récupère le code nace prédit que l'on veut expliquer

attributions = results["captum_attributions"][text_idx][top_k_idx]
# récupère les scores d'attribution captum pour la prédiction choisie
# ces scores indiquent quels tokens ont le plus influencé la prédiction

words, word_attributions = map_attributions_to_word(
    attributions.unsqueeze(0), text_sample, word_ids, offsets
)
# agrège les scores des tokens au niveau des mots
# utile car un mot peut être découpé en plusieurs tokens wordpiece

char_attributions = map_attributions_to_char(
    attributions.unsqueeze(0), offsets, text_sample
)
# projette les scores des tokens au niveau des caractères du texte

titles = [f"Attributions for NACE code {predicted_code}"]
# crée le titre des graphiques avec le code nace expliqué

figshow(plot_attributions_at_char(
    text=text_sample,
    attributions_per_char=char_attributions,
    titles=titles,
)[0])
# affiche les attributions au niveau des caractères
# permet de voir quelles parties précises du texte ont pesé dans la prédiction

figshow(plot_attributions_at_word(
    text=text_sample,
    words=words.values(),
    attributions_per_word=word_attributions,
    titles=titles,
)[0])
# affiche les attributions au niveau des mots
# permet de voir quels mots ont le plus influencé la prédiction du code nace
# %%
results_test = ttc.predict(X_test, top_k=1)
# génère une prédiction pour chaque texte du jeu test
# top_k=1 signifie qu'on garde seulement le code nace le plus probable

preds = results_test["prediction"].squeeze(1)
# récupère les codes nace prédits par le modèle
# squeeze(1) enlève une dimension inutile dans le résultat

accuracy = (preds == y_test).mean()
# compare les codes prédits aux vrais codes nace du jeu test
# calcule la part de prédictions correctes

print(f"Test accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")
# affiche l'accuracy sur le jeu test avec 4 décimales
# affiche aussi le nombre de bonnes prédictions sur le nombre total d'exemples
# %%
