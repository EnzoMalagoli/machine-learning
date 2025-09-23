import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar dataset
url = "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
df = pd.read_csv(url)

# --- Data Cleaning
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["AnnualSalary"].fillna(df["AnnualSalary"].median(), inplace=True)

# --- Encoding
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# --- Normalização
for col in ["Age", "AnnualSalary"]:
    cmin, cmax = df[col].min(), df[col].max()
    df[col] = 0.0 if cmax == cmin else (df[col] - cmin) / (cmax - cmin)

# --- Separar variáveis preditoras 
X = df[["Gender", "Age", "AnnualSalary"]]
y = df["Purchased"]

# --- Divisão em treino e teste 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Tamanho treino:", X_train.shape[0])
print("Tamanho teste:", X_test.shape[0])
