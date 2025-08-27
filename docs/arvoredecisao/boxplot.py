import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Carregar dataset
url = "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
df = pd.read_csv(url)

# --- ETAPA 1: Data Cleaning
df["AnnualSalary"].fillna(df["AnnualSalary"].median(), inplace=True)

# --- PLOT: Boxplot
fig, ax = plt.subplots(figsize=(7, 5))

bp = ax.boxplot(df["AnnualSalary"], patch_artist=True, widths=0.5)

for box in bp["boxes"]:
    box.set(facecolor="skyblue", edgecolor="navy", linewidth=1.2)
for whisker in bp["whiskers"]:
    whisker.set(color="navy", linewidth=1.2)
for cap in bp["caps"]:
    cap.set(color="navy", linewidth=1.2)
for median in bp["medians"]:
    median.set(color="darkred", linewidth=1.5)

ax.set_title("Distribuição do Salário Anual")
ax.set_ylabel("Salário Anual")
ax.set_xticks([])
ax.grid(axis="y", linestyle="--", alpha=0.6)

# Exportar como SVG
buffer = BytesIO()
plt.savefig(buffer, format="svg", bbox_inches="tight")
buffer.seek(0)
print(buffer.getvalue().decode("utf-8"))
