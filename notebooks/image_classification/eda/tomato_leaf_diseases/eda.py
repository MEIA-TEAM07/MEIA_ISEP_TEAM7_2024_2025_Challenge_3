import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('train/_classes.csv')

# Inspecionando os dados
print('-------------------Inspeção de Dados-------------------')
print(df.head())

# Verificação de dados gerais (linhas, colunas, tipos de dados e valores nulos)
print('-------------------Verificação de Dados-------------------')
print(df.info())

# Análise descritiva
print('-------------------Análise Descritiva-------------------')
print(df.describe())

# Análise por Gráfico
disease_columns = df.columns[1:]
disease_counts = df[disease_columns].sum()
plt.figure(figsize=(12,6))
disease_counts.plot(kind='bar')
plt.title('Frequência de Indicadores de Doenças')
plt.xlabel('Doença')
plt.ylabel('Número de Ocorrências')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()