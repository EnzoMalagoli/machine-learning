import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

plt.figure(figsize=(12, 10))

# ---------------- Carregar e preprocessar ----------------
url = "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
df = pd.read_csv(url)

# 1) Cleaning
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["AnnualSalary"].fillna(df["AnnualSalary"].median(), inplace=True)

# 2) Encoding (Gender -> 0/1)
enc = LabelEncoder()
df["Gender"] = enc.fit_transform(df["Gender"])  # Female=0, Male=1

# 3) Normalização Min–Max (para KNN e para o grid da fronteira)
for col in ["Age", "AnnualSalary"]:
    cmin, cmax = df[col].min(), df[col].max()
    df[col] = 0.0 if cmax == cmin else (df[col] - cmin) / (cmax - cmin)

# Features e alvo
X = df[["Gender", "Age", "AnnualSalary"]].to_numpy(dtype=float)
y = df["Purchased"].to_numpy(dtype=int)

# ---------------- Split + Treino KNN ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
knn = KNeighborsClassifier(n_neighbors=5)  # ajuste k se quiser
knn.fit(X_train, y_train)

# ---------------- Fronteira de decisão (Age x Salary) fixando Gender ----------------
GENDER_FIXED = 1  # 1=Male, 0=Female (mude aqui para ver a outra fronteira)

# grade em Age e AnnualSalary (ambos normalizados [0,1])
h = 0.01
age_min, age_max = 0.0, 1.0
sal_min, sal_max = 0.0, 1.0
xx, yy = np.meshgrid(np.arange(age_min, age_max + h, h),
                     np.arange(sal_min, sal_max + h, h))

# monta grid com Gender fixo
grid = np.c_[np.full(xx.size, GENDER_FIXED), xx.ravel(), yy.ravel()]
Z = knn.predict(grid).reshape(xx.shape)

# pontos reais do gênero escolhido
mask = (df["Gender"] == GENDER_FIXED)
Xplot = df.loc[mask, ["Age", "AnnualSalary"]].to_numpy()
yplot = df.loc[mask, "Purchased"].to_numpy()

# ---------------- Plot ----------------
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.30)
sns.scatterplot(x=Xplot[:, 0], y=Xplot[:, 1], hue=yplot, style=yplot, palette="deep", s=70)
plt.xlabel("Age (normalizado)")
plt.ylabel("AnnualSalary (normalizado)")
plt.title(f"KNN Decision Boundary (k={knn.n_neighbors}) — Gender={'Male' if GENDER_FIXED==1 else 'Female'}")
plt.legend(title="Purchased")

# ---------------- Exportar SVG para o Pages ----------------
buffer = BytesIO()
plt.savefig(buffer, format="svg", transparent=True, bbox_inches="tight")
buffer.seek(0)
print(buffer.getvalue().decode("utf-8"))
