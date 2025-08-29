import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd

from io import BytesIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# PREPROCESS 
def preprocess(df):

    # Data cleaning
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
    df["AnnualSalary"].fillna(df["AnnualSalary"].median(), inplace=True)

    # Encoding
    enc = LabelEncoder()
    df["Gender"] = enc.fit_transform(df["Gender"])  # Female=0, Male=1

    # Normalização Min–Max
    for col in ["Age", "AnnualSalary"]:
        cmin, cmax = df[col].min(), df[col].max()
        df[col] = 0.0 if cmax == cmin else (df[col] - cmin) / (cmax - cmin)

    # Features finais
    features = ["Gender", "Age", "AnnualSalary"]
    return df[features]

# CARREGAR DADOS 
url = "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
df = pd.read_csv(url)

# X (features) / y (target)
X = preprocess(df)
y = df["Purchased"]

# TRAIN / TEST SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# DECISION TREE
clf = tree.DecisionTreeClassifier(max_depth=4, random_state=42) 
clf.fit(X_train, y_train)

# AVALIAÇÃO 
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

#  PLOTAR ÁRVORE 
plt.figure(figsize=(14, 10))
tree.plot_tree(
    clf,
    feature_names=X.columns.tolist(),
    class_names=["Não comprou (0)", "Comprou (1)"],
    filled=True, rounded=True, fontsize=9
)

buf = BytesIO()
plt.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
buf.seek(0)
print(buf.getvalue().decode("utf-8"))
plt.close()
