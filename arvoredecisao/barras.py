import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Carregar dataset
url = "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
df = pd.read_csv(url)

# --- ETAPA 1: Data Cleaning 
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)


counts = df["Gender"].value_counts()

# --- PLOT: Distribuição por Gênero ---
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.bar(
    counts.index, counts.values,
    color=["pink", "skyblue"], edgecolor="lightcoral"
)

ax.set_title("Distribuição por Gênero")
ax.set_xlabel("Gênero")
ax.set_ylabel("Quantidade")
ax.grid(axis="y", linestyle="--", alpha=0.6)


buffer = BytesIO()
plt.savefig(buffer, format="svg", bbox_inches="tight")
buffer.seek(0)
print(buffer.getvalue().decode("utf-8"))
