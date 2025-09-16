import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===================== KNN (seu classificador) =====================
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # Distância Euclidiana
        distances = np.sqrt(((self.X_train - x) ** 2).sum(axis=1))
        # k vizinhos mais próximos
        k_idx = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_idx]
        # Classe mais comum
        vals, counts = np.unique(k_labels, return_counts=True)
        return vals[np.argmax(counts)]

# ===================== Pré-processamento (3 etapas) =====================
def preprocess(df):
    # 1) Data cleaning
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
    df["AnnualSalary"].fillna(df["AnnualSalary"].median(), inplace=True)

    # 2) Encoding (Gender -> 0/1)
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

    # 3) Normalização Min–Max (Age, AnnualSalary)
    for col in ["Age", "AnnualSalary"]:
        cmin, cmax = df[col].min(), df[col].max()
        df[col] = 0.0 if cmax == cmin else (df[col] - cmin) / (cmax - cmin)

    # Features finais e alvo
    X = df[["Gender", "Age", "AnnualSalary"]].to_numpy(dtype=float)
    y = df["Purchased"].to_numpy(dtype=int)
    return X, y

# Carregar dados e preparar
url = "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
df = pd.read_csv(url)
X, y = preprocess(df)

# Split treino/teste 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Treino e avaliação (KNN)
knn = KNNClassifier(k=5)  # ajuste k se quiser
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

