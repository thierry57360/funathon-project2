# https://aiml4os.github.io/funathon-project2/2-rag-intro.html

from dotenv import load_dotenv
import os
from openai import OpenAI
from qdrant_client import QdrantClient
import duckdb

# %% Load environment variables from .env file
# QDRANT_URL=https://YOURNAMESPACE-qdrant.user.lab.sspcloud.fr/
# QDRANT_API_KEY=xxxxxxxxxxxxxxxxxxxx
# QDRANT_API_PORT=443
# LLMLAB_API_KEY=xxxxxxxxxxxxxxxxxxxx
# LLMLAB_URL=https://llm.lab.sspcloud.fr/api

load_dotenv()
# %%

try:
    QDRANT_URL = os.environ["QDRANT_URL"]
    print("QDRANT_URL loaded successfully")
except KeyError:
    raise ValueError("QDRANT_URL is not set — check your .env file")

# %% Conncet to LLM API

client_llmlab = OpenAI(
    base_url=os.environ["LLMLAB_URL"],
    api_key=os.environ["LLMLAB_API_KEY"],
)

# Print models list
models = client_llmlab.models.list()
for model in models.data:
    print(f"ID: {model.id}")

# %% Connet to qdrant

client_qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    port=os.environ["QDRANT_API_PORT"],
    check_compatibility=False
)

collections = client_qdrant.get_collections()
for collection in collections.collections:
    print(collection.name)

# %% Import NACE data as a list of dictionaries
con = duckdb.connect(database=":memory:")

con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

path_nace = 'https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/NACE_Rev2.1_Structure_Explanatory_Notes_EN.tsv'  # noqa: E501
query_definition = f"SELECT * FROM read_csv('{path_nace}')"
table = con.execute(query_definition).to_arrow_table()
nace = table.to_pylist()

# %%
nace[22]
# %%
