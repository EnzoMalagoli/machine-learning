import random
import pandas as pd
from collections import Counter

# ===== PREPROCESS =====
def preprocess(df):
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
    df["AnnualSalary"].fillna(df["AnnualSalary"].median(), inplace=True)

    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

    for col in ["Age", "AnnualSalary"]:
        cmin, cmax = df[col].min(), df[col].max()
        df[col] = 0.0 if cmax == cmin else (df[col] - cmin) / (cmax - cmin)

    X = df[["Gender", "Age", "AnnualSalary"]].values.tolist()
    y = df["Purchased"].astype(int).tolist()
    return X, y

# ===== LOAD DATA =====
url = "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
df = pd.read_csv(url)
X, y = preprocess(df)

# ===== STRATIFIED SPLIT (listas, sem numpy) =====
def stratified_train_test_split(X, y, test_size=0.3, seed=42):
    rnd = random.Random(seed)
    idx0 = [i for i, t in enumerate(y) if t == 0]
    idx1 = [i for i, t in enumerate(y) if t == 1]
    rnd.shuffle(idx0); rnd.shuffle(idx1)
    n0_test = max(1, int(len(idx0) * test_size))
    n1_test = max(1, int(len(idx1) * test_size))
    test_idx = idx0[:n0_test] + idx1[:n1_test]
    train_idx = idx0[n0_test:] + idx1[n1_test:]
    def take(ix): 
        return [X[i] for i in ix], [y[i] for i in ix]
    return *take(train_idx), *take(test_idx)

X_train, y_train, X_test, y_test = stratified_train_test_split(X, y, test_size=0.3, seed=42)

# ===== GINI, TREE, FOREST (seu estilo) =====
def gini_impurity(y):
    if not y:
        return 0.0
    counts = Counter(y)
    imp = 1.0
    n = len(y)
    for c in counts.values():
        p = c / n
        imp -= p * p
    return imp

def split_dataset(X, y, feature_idx, value):
    left_X, left_y, right_X, right_y = [], [], [], []
    for i in range(len(X)):
        if X[i][feature_idx] <= value:
            left_X.append(X[i]); left_y.append(y[i])
        else:
            right_X.append(X[i]); right_y.append(y[i])
    return left_X, left_y, right_X, right_y

class Node:
    def __init__(self, feature_idx=None, value=None, left=None, right=None, label=None):
        self.feature_idx = feature_idx
        self.value = value
        self.left = left
        self.right = right
        self.label = label

def build_tree(X, y, max_depth, min_samples_split, max_features, rnd):
    if len(y) < min_samples_split or max_depth == 0:
        return Node(label=Counter(y).most_common(1)[0][0])

    n_features = len(X[0])
    features = rnd.sample(range(n_features), max_features)

    best_gini = float("inf")
    best = None

    for f in features:
        values = sorted({row[f] for row in X})
        for v in values:
            LX, Ly, RX, Ry = split_dataset(X, y, f, v)
            if not Ly or not Ry:
                continue
            pL = len(Ly) / len(y)
            g = pL * gini_impurity(Ly) + (1 - pL) * gini_impurity(Ry)
            if g < best_gini:
                best_gini = g
                best = (f, v, LX, Ly, RX, Ry)

    if best is None:
        return Node(label=Counter(y).most_common(1)[0][0])

    f, v, LX, Ly, RX, Ry = best
    left = build_tree(LX, Ly, max_depth - 1, min_samples_split, max_features, rnd)
    right = build_tree(RX, Ry, max_depth - 1, min_samples_split, max_features, rnd)
    return Node(f, v, left, right)

def predict_tree(node, x):
    if node.label is not None:
        return node.label
    if x[node.feature_idx] <= node.value:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)

class RandomForest:
    def __init__(self, n_estimators=25, max_depth=6, min_samples_split=2, max_features="sqrt", seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        rnd = random.Random(self.seed)
        n = len(y)
        n_features = len(X[0])
        mf = int(n_features ** 0.5) if self.max_features == "sqrt" else self.max_features

        for _ in range(self.n_estimators):
            idx = [rnd.randint(0, n - 1) for _ in range(n)]
            Xb = [X[i] for i in idx]
            yb = [y[i] for i in idx]
            tree = build_tree(Xb, yb, self.max_depth, self.min_samples_split, mf, rnd)
            self.trees.append(tree)

    def predict(self, X):
        preds = []
        for x in X:
            votes = [predict_tree(t, x) for t in self.trees]
            preds.append(Counter(votes).most_common(1)[0][0])
        return preds

rf = RandomForest(n_estimators=50, max_depth=6, min_samples_split=2, max_features="sqrt", seed=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = sum(1 for yp, yt in zip(y_pred, y_test) if yp == yt) / len(y_test)
print(f"Accuracy: {acc:.2f}")

cm = [[0, 0], [0, 0]]
for yt, yp in zip(y_test, y_pred):
    cm[yt][yp] += 1
cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
print("\nConfusion Matrix:")
print(cm_df.to_markdown())
