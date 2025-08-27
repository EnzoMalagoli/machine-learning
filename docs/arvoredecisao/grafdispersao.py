import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Carregar dataset
url = "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
df = pd.read_csv(url)

# --- ETAPA 1: Data Cleaning ---
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["AnnualSalary"].fillna(df["AnnualSalary"].median(), inplace=True)

# --- ETAPA 2: Encoding (Gender -> numérico) ---
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# --- ETAPA 3: Normalização (min–max) ---
for col in ["Age", "AnnualSalary"]:
    cmin, cmax = df[col].min(), df[col].max()
    df[col] = 0.0 if cmax == cmin else (df[col] - cmin) / (cmax - cmin)

# Separar classes
df0 = df[df["Purchased"] == 0]
df1 = df[df["Purchased"] == 1]

# --- PLOT: Dispersão Idade x Salário ---
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

ax.scatter(
    df0["Age"], df0["AnnualSalary"],
    label="Não comprou (0)", alpha=0.4,
    color="lightcoral", edgecolor="darkred", linewidth=0.8
)
ax.scatter(
    df1["Age"], df1["AnnualSalary"],
    label="Comprou (1)", alpha=0.4,
    color="skyblue", edgecolor="navy", linewidth=0.8
)

ax.set_title("Idade x Salário por Decisão de Compra")
ax.set_xlabel("Idade")
ax.set_ylabel("Salário Anual")
ax.grid(linestyle="--", alpha=0.6)
ax.legend()

# Exportar SVG em buffer (modelo que você usa)
buffer = BytesIO()
plt.savefig(buffer, format="svg", bbox_inches="tight")
buffer.seek(0)
print(buffer.getvalue().decode("utf-8"))
