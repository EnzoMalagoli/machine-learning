import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

url = "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
df = pd.read_csv(url)

df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

X_all = df[["Gender", "Age", "AnnualSalary"]].values.astype(float)
y_raw = df["Purchased"].values
y_all = np.where(y_raw == 1, 1, -1)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
)

def rbf_kernel(x1, x2, sigma=1.0):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))

def kernel_matrix(X, kernel, sigma):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(X[i], X[j], sigma)
    return K

sigma = 1.0
K_train = kernel_matrix(X_train, rbf_kernel, sigma)

n_train = len(y_train)
P = np.outer(y_train, y_train) * K_train

def objective(alpha):
    return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)

def constraint(alpha):
    return np.dot(alpha, y_train)

cons = {"type": "eq", "fun": constraint}
bounds = [(0, None) for _ in range(n_train)]
alpha0 = np.zeros(n_train)

res = optimize.minimize(
    objective,
    alpha0,
    method="SLSQP",
    bounds=bounds,
    constraints=cons,
    options={"maxiter": 1000}
)

alpha = res.x

sv_threshold = 1e-5
sv_idx = alpha > sv_threshold
sv_indices = np.where(sv_idx)[0]

i = sv_indices[0]
b = y_train[i] - np.dot(alpha * y_train, K_train[i, :])

def predict_one(x):
    kx = np.array([rbf_kernel(x, xi, sigma=sigma) for xi in X_train])
    return np.dot(alpha * y_train, kx) + b

def predict_batch(X):
    scores = np.array([predict_one(x) for x in X])
    return np.where(scores >= 0, 1, -1)

y_pred_test = predict_batch(X_test)

acc = accuracy_score(y_test, y_pred_test)
cm = confusion_matrix(y_test, y_pred_test, labels=[-1, 1])

print("Accuracy (teste):", acc)
print("Confusion matrix (linhas = verdade, colunas = predito, ordem: -1, +1)")
print(cm)

y_test_01 = np.where(y_test == 1, 1, 0)
y_pred_01 = np.where(y_pred_test == 1, 1, 0)
cm_01 = confusion_matrix(y_test_01, y_pred_01, labels=[0, 1])

print("\nMatriz de confusão em termos de Purchased (0/1):")
print(cm_01)
