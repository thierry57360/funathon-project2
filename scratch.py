# %%
print("cellule 1 OK")

# %%
import pandas as pd
import matplotlib.pyplot as plt
print("imports OK")

# %%
url = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"

df = pd.read_parquet(url)

print("chargement OK")
print(df.head())
print(df.shape)
print(df.columns)

# %%
df["code"].value_counts().head(10).plot(kind="bar")
plt.tight_layout()
plt.show()

print("graphique OK")
# %%
