import pandas as pd

def preprocess(df):
    # Limpeza
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['AnnualSalary'].fillna(df['AnnualSalary'].median(), inplace=True)

    # Encoding simples para Gender
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Retorna todas as colunas (sem cortar nada)
    return df

# Carregar dataset
df = pd.read_csv('https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv')
df = df.sample(n=10, random_state=42)
df = preprocess(df)

# Exibir todas as colunas no GitHub Pages
print(df.to_markdown(index=False))
