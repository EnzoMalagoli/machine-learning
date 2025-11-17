import numpy as np
import pandas as pd
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def preprocess(df):
   
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
    df["AnnualSalary"].fillna(df["AnnualSalary"].median(), inplace=True)
    
    enc = LabelEncoder()
    df["Gender"] = enc.fit_transform(df["Gender"]) 
   
    for col in ["Age", "AnnualSalary"]:
        cmin, cmax = df[col].min(), df[col].max()
        df[col] = 0.0 if cmax == cmin else (df[col] - cmin) / (cmax - cmin)
    # X / y
    X = df[["Gender", "Age", "AnnualSalary"]].to_numpy(float)
    y = df["Purchased"].to_numpy(int)
    return X, y


df = pd.read_csv("https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv")
X, y = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)


knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(pd.DataFrame(cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"]).to_markdown())
