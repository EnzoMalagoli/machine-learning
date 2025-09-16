import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#  PREPROCESS 
def preprocess(df):

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
    df["AnnualSalary"].fillna(df["AnnualSalary"].median(), inplace=True)


    enc = LabelEncoder()
    df["Gender"] = enc.fit_transform(df["Gender"])


    for col in ["Age", "AnnualSalary"]:
        cmin, cmax = df[col].min(), df[col].max()
        df[col] = 0.0 if cmax == cmin else (df[col] - cmin) / (cmax - cmin)

    X = df[["Gender", "Age", "AnnualSalary"]].to_numpy(dtype=float)
    y = df["Purchased"].to_numpy(dtype=int)
    return X, y

#  LOAD DATA 
url = "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
df = pd.read_csv(url)
X, y = preprocess(df)

#  SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# TRAIN KNN 
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)  # k=5, distância Euclidiana
knn.fit(X_train, y_train)

# PREDICT & EVALUATE
y_pred = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


