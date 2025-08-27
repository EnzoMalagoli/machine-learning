import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Carregar dataset
url = "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
df = pd.read_csv(url)

# Separar quem comprou e quem não comprou
df0 = df[df["Purchased"] == 0]
df1 = df[df["Purchased"] == 1]

# --- PLOT: Dispersão Idade x Salário ---
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

ax.scatter(df0["Age"], df0["AnnualSalary"], 
           label="Não comprou (0)", 
           alpha=0.7, color="lightcoral", edgecolor="darkred")

ax.scatter(df1["Age"], df1["AnnualSalary"], 
           label="Comprou (1)", 
           alpha=0.7, color="skyblue", edgecolor="navy")

ax.set_title("Idade x Salário por Decisão de Compra")
ax.set_xlabel("Idade")
ax.set_ylabel("Salário Anual")
ax.grid(linestyle="--", alpha=0.6)
ax.legend()

# Salvar em buffer como SVG (para Pages)
buffer = BytesIO()
plt.savefig(buffer, format="svg", bbox_inches="tight")
buffer.seek(0)
print(buffer.getvalue().decode("utf-8"))
