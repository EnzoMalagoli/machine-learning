import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans

plt.figure(figsize=(12, 10))


url = "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
df = pd.read_csv(url)


X = df[["Age", "AnnualSalary"]].dropna().to_numpy()


kmeans = KMeans(n_clusters=2, init="k-means++", max_iter=100, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)


plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="red", marker="*", s=200, label="Centroides")

plt.title("Clusters com K-Means")
plt.xlabel("Idade")
plt.ylabel("Salário Anual")
plt.legend()


buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
