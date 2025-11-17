import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Carregar dataset
url = "https://raw.githubusercontent.com/EnzoMalagoli/machine-learning/refs/heads/main/data/car_data.csv"
df = pd.read_csv(url)

# Selecionar colunas numéricas para normalizar
features_to_normalize = ['Age', 'AnnualSalary']

# Inicializar o scaler
scaler = MinMaxScaler()

# Aplicar normalização e substituir no DataFrame
df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

# Mostrar amostra dos dados normalizados
print(df.sample(10).to_markdown(index=False))
