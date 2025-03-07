import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('notebooks/image_classification/eda/tomato_leaf_diseases/dataset/train/_classes.csv')

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


plt.figure(figsize=(10,6))
plt.bar(disease_counts.index, disease_counts.values, width=1.0, edgecolor='black')
plt.title('Frequência de Indicadores de Doenças')
plt.xlabel('Doenças')
plt.ylabel('Número de Ocorrências')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()