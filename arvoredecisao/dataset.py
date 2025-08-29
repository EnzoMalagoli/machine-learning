import pandas as pd

def preprocess(df):
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
    df["AnnualSalary"].fillna(df["AnnualSalary"].median(), inplace=True)

    features = ["Gender", "Age", "AnnualSalary", "Purchased"]
    return df[features]

# Carregar dataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
)


df = df.sample(n=10, random_state=42)
df = preprocess(df)


print(df.to_markdown(index=False))
