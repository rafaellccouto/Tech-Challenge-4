# Análise Exploratória Atualizada (reconstruída a partir do pipeline)

Este documento contém a Análise Exploratória de Dados (EDA) reconstruída com base no novo pipeline presente em `01_ML_Pipeline_Obesity_Prediction.ipynb`. Inclui os trechos de código e descrições necessários para reproduzir as visualizações e estatísticas.

Observação: execute as células do notebook ou os trechos de código abaixo para gerar gráficos e tabelas localmente — o repositório contém os dados em [Dados_base/Obesity.csv](Dados_base/Obesity.csv).

## 1. Preparação e Carregamento

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

dados_path = 'Dados_base/Obesity.csv'
df = pd.read_csv(dados_path)
print('Dimensões:', df.shape)
display(df.head())
```

## 2. Inspeção inicial

```python
# Tipos, missing values e estatísticas descritivas
print(df.dtypes)
print(df.isnull().sum())
display(df.describe())

# Distribuição da variável alvo
print(df['Obesity'].value_counts())
print((df['Obesity'].value_counts(normalize=True) * 100).round(2))
```

## 3. Visualizações principais

```python
# Distribuição de classes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['Obesity'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Distribuição de Classes de Obesidade')
df['Obesity'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
axes[1].set_ylabel('')
plt.tight_layout()
plt.show()

# Distribuições numéricas exemplares
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes[0,0].hist(df['Age'], bins=30); axes[0,0].set_title('Age')
axes[0,1].hist(df['Height'], bins=30); axes[0,1].set_title('Height')
axes[0,2].hist(df['Weight'], bins=30); axes[0,2].set_title('Weight')
axes[1,0].hist(df['FAF'], bins=20); axes[1,0].set_title('FAF')
axes[1,1].hist(df['CH2O'], bins=20); axes[1,1].set_title('CH2O')
axes[1,2].hist(df['TUE'], bins=20); axes[1,2].set_title('TUE')
plt.tight_layout(); plt.show()
```

## 4. Análise de variáveis categóricas

```python
categorical_cols = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
for col in categorical_cols:
	print(f"{col}:\n", df[col].value_counts(), "\n")
```

## 5. Engenharia de features (resumo aplicado no pipeline)

- Mantivemos variáveis de comportamento como `FCVC`, `NCP`, `CH2O`, `FAF`, `TUE` em precisão decimal.
- Criamos features derivadas: `Snack_Health_Score` (mapeamento de `CAEC`), `Lifestyle_Balance` (`FAF - TUE`) e `Water_per_Meal` (`CH2O / (NCP + 1)`).
- Removemos colunas antropométricas (`Height`, `Weight`, `BMI`) do conjunto de treino para focar em sinais comportamentais.

Trecho de geração de features:

```python
caec_map = {'no': 4, 'Sometimes': 3, 'Frequently': 2, 'Always': 1}
df['Snack_Health_Score'] = df['CAEC'].map(caec_map)
df['Lifestyle_Balance'] = df['FAF'] - df['TUE']
df['Water_per_Meal'] = df['CH2O'] / (df['NCP'] + 1)
```

## 6. Encoding e preparação final

- Binarização: `family_history`, `FAVC`, `SMOKE`, `SCC` → 0/1
- `Gender`: Label encoding (Male=1)
- One-hot para `CAEC`, `CALC`, `MTRANS`

```python
binary_cols = ['family_history', 'FAVC', 'SMOKE', 'SCC']
for col in binary_cols:
	df[col] = (df[col] == 'yes').astype(int)
df['Gender'] = (df['Gender'] == 'Male').astype(int)
df_final = pd.get_dummies(df.drop(columns=['Height','Weight','BMI'], errors='ignore'), columns=['CAEC','CALC','MTRANS'], drop_first=False)
```

## 7. Correlações e insights rápidos

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_corr = df_final.copy()
df_corr['Obesity_encoded'] = le.fit_transform(df_corr['Obesity'])
corrs = df_corr.corr()['Obesity_encoded'].abs().sort_values(ascending=False)
display(corrs.head(11))
```

## 8. Como reproduzir (sugestão rápida)

1. Abra `01_ML_Pipeline_Obesity_Prediction.ipynb` e execute as células em ordem (estágios 1–3 geram EDA e features).
2. Se preferir, execute os trechos acima em um script Python para gerar as figuras em lote.
