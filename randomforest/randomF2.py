import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ===== PREPROCESS =====
def preprocess(df):
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
    df["AnnualSalary"].fillna(df["AnnualSalary"].median(), inplace=True)

    enc = LabelEncoder()
    df["Gender"] = enc.fit_transform(df["Gender"])  # Female=0, Male=1

    for col in ["Age", "AnnualSalary"]:
        cmin, cmax = df[col].min(), df[col].max()
        df[col] = 0.0 if cmax == cmin else (df[col] - cmin) / (cmax - cmin)

    X = df[["Gender", "Age", "AnnualSalary"]].to_numpy(float)
    y = df["Purchased"].to_numpy(int)
    return X, y, ["Gender", "Age", "AnnualSalary"]

# ===== DATA =====
url = "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
df = pd.read_csv(url)
X, y, feat_names = preprocess(df)

# ===== SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# ===== MODEL =====
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# ===== EVAL =====
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")
print(f"OOB Score: {getattr(rf, 'oob_score_', float('nan')):.2f}")

cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
print("\nConfusion Matrix:")
print(cm_df.to_markdown())

imp = pd.DataFrame({"Feature": feat_names, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)
print("\nFeature Importances:")
print(imp.to_markdown(index=False))
