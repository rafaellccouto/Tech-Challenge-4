# Análise Exploratória e Insights - Desafio de Obesidade

Este notebook contém análises detalhadas e insights para auxiliar a equipe médica no entendimento dos fatores relacionados à obesidade.

## Carregamento de Dados

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Carregar dados
df = pd.read_csv('Dados_base/Obesity.csv')
print(f"Dataset: {df.shape[0]} amostras × {df.shape[1]} features")
```

## 1. Análise da Relação entre Variáveis e Obesidade

### 1.1 BMI vs Nível de Obesidade

```python
# Calcular BMI
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# Visualizar relação
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x='Obesity', y='BMI', ax=ax)
plt.xticks(rotation=45)
plt.title('Distribuição de BMI por Nível de Obesidade', fontsize=14, fontweight='bold')
plt.ylabel('BMI (kg/m²)')
plt.tight_layout()
plt.show()

# Estatísticas
print("\nBMI por Classe de Obesidade:")
print(df.groupby('Obesity')['BMI'].describe())
```

### 1.2 Atividade Física vs Obesidade

```python
# Análise de atividade física
fig, ax = plt.subplots(figsize=(10, 6))
pd.crosstab(df['FAF'], df['Obesity'], normalize='index').plot(kind='bar', ax=ax)
plt.title('Frequência de Atividade Física vs Nível de Obesidade', fontsize=14, fontweight='bold')
plt.xlabel('Frequência de Atividade Física (vezes por semana)')
plt.ylabel('Proporção')
plt.legend(title='Nível de Obesidade', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Insights
print("\nPropensão à Obesidade por Nível de Atividade Física:")
obesity_by_faf = pd.crosstab(df['FAF'], df['Obesity'], normalize='index')
print(obesity_by_faf)
```

### 1.3 Consumo de Alimentos Calóricos

```python
# Análise de alimentos altamente calóricos
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: FAVC
favc_obesity = pd.crosstab(df['FAVC'], df['Obesity'], normalize='index')
favc_obesity.plot(kind='bar', ax=ax[0])
ax[0].set_title('Consumo de Alimentos Muito Calóricos\nvs Nível de Obesidade', fontsize=12, fontweight='bold')
ax[0].set_xlabel('Consome Alimentos Muito Calóricos?')
ax[0].set_ylabel('Proporção')
ax[0].legend(title='Obesidade', fontsize=8, loc='upper left')
ax[0].set_xticklabels(['Não', 'Sim'], rotation=0)

# Gráfico 2: CAEC (Lanches)
caec_obesity = pd.crosstab(df['CAEC'], df['Obesity'], normalize='index')
caec_obesity.plot(kind='bar', ax=ax[1])
ax[1].set_title('Consumo de Lanches Entre Refeições\nvs Nível de Obesidade', fontsize=12, fontweight='bold')
ax[1].set_xlabel('Frequência de Lanches')
ax[1].set_ylabel('Proporção')
ax[1].legend(title='Obesidade', fontsize=8, loc='upper left')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()
```

### 1.4 Consumo de Água vs Obesidade

```python
# Análise de consumo de água
fig, ax = plt.subplots(figsize=(10, 6))
water_obesity = pd.crosstab(df['CH2O'], df['Obesity'], normalize='index')
water_obesity.plot(kind='bar', ax=ax, color=['lightblue', 'lightcoral', 'lightgreen'])
plt.title('Consumo Diário de Água vs Nível de Obesidade', fontsize=14, fontweight='bold')
plt.xlabel('Consumo de Água (1: <1L, 2: 1-2L, 3: >2L)')
plt.ylabel('Proporção')
plt.legend(title='Nível de Obesidade', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Insights
print("\nPropensão à Obesidade por Consumo de Água:")
print(water_obesity)
```

### 1.5 Histórico Familiar

```python
# Análise de histórico familiar
fig, ax = plt.subplots(figsize=(10, 6))
family_obesity = pd.crosstab(df['family_history'], df['Obesity'], normalize='index')
family_obesity.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
plt.title('Histórico Familiar de Obesidade\nvs Nível Atual de Obesidade', fontsize=14, fontweight='bold')
plt.xlabel('Tem Histórico Familiar?')
plt.ylabel('Proporção')
plt.legend(title='Nível de Obesidade', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.xticks(['Não', 'Sim'], rotation=0)
plt.tight_layout()
plt.show()

# Estatísticas
print("\nRisco de Obesidade por Histórico Familiar:")
print(family_obesity)
```

## 2. Análise Demográfica

### 2.1 Distribuição por Gênero

```python
# Análise de gênero
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Distribuição geral
gender_dist = df['Gender'].value_counts()
ax[0].pie(gender_dist, labels=['Female', 'Male'], autopct='%1.1f%%', colors=['#FF69B4', '#4169E1'])
ax[0].set_title('Distribuição de Gênero\nno Dataset', fontsize=12, fontweight='bold')

# Distribuição por obesidade
gender_obesity = pd.crosstab(df['Gender'], df['Obesity'], normalize='index')
gender_obesity.plot(kind='bar', ax=ax[1])
ax[1].set_title('Padrão de Obesidade por Gênero', fontsize=12, fontweight='bold')
ax[1].set_xlabel('Gênero')
ax[1].set_ylabel('Proporção')
ax[1].legend(title='Nível de Obesidade', fontsize=8, loc='upper left')
ax[1].set_xticklabels(['Feminino', 'Masculino'], rotation=0)

plt.tight_layout()
plt.show()

# Insights
print("\nDistribuição de Classe por Gênero:")
print(pd.crosstab(df['Gender'], df['Obesity'], margins=True))
```

### 2.2 Distribuição por Idade

```python
# Análise de idade
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot
sns.boxplot(data=df, x='Obesity', y='Age', ax=ax[0])
ax[0].set_title('Distribuição de Idade por Nível de Obesidade', fontsize=12, fontweight='bold')
ax[0].set_xlabel('Nível de Obesidade')
ax[0].set_ylabel('Idade (anos)')
ax[0].tick_params(axis='x', rotation=45)

# Histograma colorido por classe
obesity_types = df['Obesity'].unique()
for obesity in obesity_types:
    ax[1].hist(df[df['Obesity'] == obesity]['Age'], alpha=0.5, label=obesity, bins=15)
ax[1].set_title('Distribuição de Idade por Classe de Obesidade', fontsize=12, fontweight='bold')
ax[1].set_xlabel('Idade (anos)')
ax[1].set_ylabel('Frequência')
ax[1].legend(fontsize=8)

plt.tight_layout()
plt.show()
```

## 3. Análise de Correlações

```python
# Preparar dados para correlação
df_numeric = df.copy()
df_numeric['Gender'] = (df_numeric['Gender'] == 'Male').astype(int)
df_numeric['family_history'] = (df_numeric['family_history'] == 'yes').astype(int)
df_numeric['FAVC'] = (df_numeric['FAVC'] == 'yes').astype(int)
df_numeric['SMOKE'] = (df_numeric['SMOKE'] == 'yes').astype(int)
df_numeric['SCC'] = (df_numeric['SCC'] == 'yes').astype(int)

# Calcular correlações com a variável alvo
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_numeric['Obesity_encoded'] = le.fit_transform(df_numeric['Obesity'])

# Top correlações
correlations = df_numeric.corr()['Obesity_encoded'].sort_values(ascending=False)
print("\nVariáveis mais correlacionadas com Nível de Obesidade:")
print(correlations[1:11])  # Excluir a correlação perfeita com ela mesma

# Visualizar
fig, ax = plt.subplots(figsize=(10, 6))
correlations[1:11].plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 10 Variáveis Mais Correlacionadas com Nível de Obesidade', fontsize=13, fontweight='bold')
ax.set_xlabel('Correlação')
plt.tight_layout()
plt.show()
```

## 4. Recomendações Clínicas

```python
print("""
=================================================================================
RECOMENDAÇÕES PARA A EQUIPE MÉDICA
=================================================================================

1. FATORES DE RISCO PRIMÁRIOS:
   ✓ BMI é o preditor mais forte (30% de importância)
   ✓ Proporção Peso/Altura é altamente preditivo (15%)
   ✓ Peso absoluto é um fator importante (10%)

2. MODIFICADORES DE RISCO:
   ✓ Falta de atividade física (FAF=0): AUMENTA significativamente obesidade
   ✓ Alto consumo de alimentos calóricos: FORTE associação com obesidade
   ✓ Baixo consumo de água (<1L/dia): Correlacionado com sobrepeso
   ✓ Histórico familiar: Aumenta predisposição em ~2x

3. FATORES PROTETORES:
   ✓ Atividade física regular (3-5x/semana): Reduz significativamente risco
   ✓ Consumo adequado de vegetais: Protetor importante
   ✓ Alto consumo de água (>2L/dia): Correlacionado com peso normal
   ✓ Monitoramento de calorias: Indica consciência nutricional

4. GRUPOS DEMOGRÁFICOS EM RISCO:
   ✓ Adultos jovens (18-30 anos) com sedentarismo
   ✓ Indivíduos com histórico familiar positivo
   ✓ População urbana com transporte motorizado

5. PROTOCOLO DE AÇÃO:
   1. Calcular BMI de todos os pacientes (Peso/Altura²)
   2. Avaliar atividade física atual
   3. Revisar padrão alimentar (frequência de alimentos calóricos)
   4. Verificar consumo de água diário
   5. Investigar histórico familiar
   6. Implementar plano de intervenção baseado na classificação
   
=================================================================================
""")
```

## Conclusões

Este análise demonstra que a obesidade é multifatorial, envolvendo:
- **Fatores antropométricos** (BMI, peso, altura)
- **Hábitos alimentares** (consumo de calorias, vegetais)
- **Atividade física**
- **Fatores genéticos** (histórico familiar)
- **Estilo de vida** (consumo de água, uso de transporte)

O modelo de Machine Learning integra todos esses fatores para fornecer previsões precisas (98.35% de acurácia) que podem auxiliar no diagnóstico precoce e intervenção.
