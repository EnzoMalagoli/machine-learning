import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score


url = "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
df = pd.read_csv(url)


df["Age"].fillna(df["Age"].median(), inplace=True)
df["AnnualSalary"].fillna(df["AnnualSalary"].median(), inplace=True)
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)


enc = LabelEncoder()
df["Gender"] = enc.fit_transform(df["Gender"])


X = df[["Age", "AnnualSalary"]].values
y = df["Purchased"].values


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")


h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)


for class_value in np.unique(y):
    plt.scatter(
        X[y == class_value, 0],
        X[y == class_value, 1],
        label=f"Classe {class_value}",
        edgecolor="k"
    )

plt.xlabel("Idade (normalizada)")
plt.ylabel("Salário Anual (normalizado)")
plt.title("KNN Decision Boundary (k=3)")
plt.legend()


buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
