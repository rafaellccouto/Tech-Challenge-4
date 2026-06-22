# 🏥 Sistema de Previsão de Obesidade - Tech Challenge Postech

## 📋 Descrição do Projeto

Sistema de Machine Learning para auxiliar médicos e médicas na previsão e diagnóstico de obesity. O modelo utiliza dados antropométricos, alimentares e de estilo de vida para classificar o nível de obesidade em 7 categorias.

**Acurácia: 98.35%** ✅ (Acima do target de 75%)

## 🎯 Objetivos Alcançados

- ✅ **Pipeline completo de ML** com feature engineering e treinamento
- ✅ **Modelo com 98.35% de acurácia** (Gradient Boosting)
- ✅ **Deploy em Streamlit** - Aplicação interativa
- ✅ **Análise exploratória** com insights para equipe médica
- ✅ **Código documentado** e compartilhado no GitHub

## 📊 Estrutura do Projeto

```
Tech Challenge 4/
├── 01_ML_Pipeline_Obesity_Prediction.ipynb  # Notebook com pipeline completo
├── 02_Analise_Exploratoria_Insights.md      # Insights analíticos
├── streamlit_app.py                          # Aplicação Streamlit
├── models/                                   # Modelos treinados
│   ├── best_obesity_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
├── Dados_base/
│   └── Obesity.csv                          # Dataset original
└── README.md                                # Este arquivo
```

## 🚀 Como Executar

### Pré-requisitos

- Python 3.8+
- pip ou conda

### 1. Instalar Dependências

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib scipy
```

### 2. Treinar o Modelo (Opcional)

Se quiser retreinar o modelo:

```bash
jupyter notebook 01_ML_Pipeline_Obesity_Prediction.ipynb
```

Então execute todas as células. Os modelos serão salvos em `./models/`

### 3. Executar a Aplicação Streamlit

```bash
streamlit run streamlit_app.py
```

A aplicação abrirá em `http://localhost:8501`

## 📈 Resultados do Modelo

### Performance Geral

| Métrica | Valor |
|---------|-------|
| **Acurácia** | **98.35%** |
| Precisão (média) | 98.37% |
| Recall (média) | 98.35% |
| F1-Score (média) | 98.34% |

### Comparação de Modelos Testados

| Modelo | Acurácia |
|--------|----------|
| Ridge Logistic Regression | 94.80% |
| Random Forest | 97.87% |
| **Gradient Boosting** | **98.35%** ✅ |
| Voting Ensemble | 95.51% |

### Distribuição de Classes

Dataset contém 2.111 amostras distribuídas em 7 classes:

- Obesity_Type_I: 351 amostras (16.6%)
- Obesity_Type_III: 324 amostras (15.3%)
- Obesity_Type_II: 297 amostras (14.1%)
- Overweight_Level_I: 290 amostras (13.7%)
- Overweight_Level_II: 290 amostras (13.7%)
- Normal_Weight: 287 amostras (13.6%)
- Insufficient_Weight: 272 amostras (12.9%)

## 🔍 Dicionário de Dados

### Variáveis de Entrada

| Variável | Descrição | Valores |
|----------|-----------|---------|
| Gender | Gênero | Female, Male |
| Age | Idade | 14-61 anos |
| Height | Altura | 1.45-1.98 metros |
| Weight | Peso | 39-173 kg |
| family_history | Histórico familiar de obesidade | yes, no |
| FAVC | Consome alimentos muito calóricos | yes, no |
| FCVC | Frequência de vegetais | 1-3 (raramente a sempre) |
| NCP | Refeições principais/dia | 1-4 |
| CAEC | Lanches entre refeições | no, Sometimes, Frequently, Always |
| SMOKE | Fuma | yes, no |
| CH2O | Consumo de água diário | 1-3 (<1L a >2L) |
| SCC | Monitora calorias | yes, no |
| FAF | Atividade física/semana | 0-3 (nenhuma a 5x) |
| TUE | Tempo com eletrônicos | 0-2 (0-2h a >5h) |
| CALC | Consumo de álcool | no, Sometimes, Frequently, Always |
| MTRANS | Meio de transporte | Automobile, Motorbike, Bike, Public_Transportation, Walking |

### Variável Alvo

| Classe | Descrição |
|--------|-----------|
| Insufficient_Weight | Abaixo do peso |
| Normal_Weight | Peso normal |
| Overweight_Level_I | Sobrepeso Nível I |
| Overweight_Level_II | Sobrepeso Nível II |
| Obesity_Type_I | Obesidade Tipo I |
| Obesity_Type_II | Obesidade Tipo II |
| Obesity_Type_III | Obesidade Tipo III (mórbida) |

## 💡 Features Mais Importantes

As variáveis com maior impacto na previsão (segundo Random Forest):

1. **BMI** (~30%) - Índice de Massa Corporal
2. **Weight_Height_Ratio** (~15%) - Proporção Peso/Altura
3. **Weight** (~10%) - Peso absoluto
4. **Gender** (~8%) - Gênero
5. **Age_Family_History** (~6%) - Interação Idade × Histórico Familiar
6. **Height** (~5%) - Altura
7. **Age** (~4%) - Idade

## 🔬 Metodologia

### Fases da Pipeline

#### 1. Exploração de Dados (EDA)
- Análise de distribuição de classes
- Verificação de dados faltantes (0 faltantes)
- Análise estatística descritiva
- Visualizações exploratórias

#### 2. Preparação de Dados
- Limpeza: Arredondamento de variáveis com ruído decimal
- Encoding: 
  - Label encoding para variáveis binárias
  - One-Hot encoding para multicategoriais
- Feature Engineering:
  - BMI = Peso / (Altura²)
  - FAF_Weight = FAF × Weight
  - Age_Family_History = Age × family_history
  - Weight_Height_Ratio = Weight / Height

#### 3. Modelagem
- Train-Test Split: 80% treino, 20% teste (estratificado)
- Normalização: StandardScaler
- Validação: 5-fold Cross-Validation
- Hiperparametrização: GridSearchCV

#### 4. Modelos Testados
- Ridge Logistic Regression (L2 Regularization)
- Random Forest (200 estimadores)
- Gradient Boosting (200 estimadores, LR=0.1)
- Voting Classifier (ensemble)

#### 5. Seleção do Modelo
**Gradient Boosting** selecionado por melhor desempenho (98.35%)

## 🎨 Interface da Aplicação Streamlit

A aplicação oferece 4 abas principais:

### 1. 🔮 Previsão
- Formulário interativo para entrada de dados
- Cálculo automático de BMI
- Previsão em tempo real
- Visualização de probabilidades
- Recomendações personalizadas de saúde

### 2. 📊 Análise de Dados
- Distribuição de classes
- Features mais importantes
- Estatísticas descritivas

### 3. ℹ️ Sobre o Modelo
- Algoritmo utilizado
- Métricas de desempenho
- Comparação entre modelos
- Requisitos atendidos

### 4. 📖 Dicionário
- Descrição de todas as variáveis
- Classificações de obesidade
- Referência rápida

## 📚 Insights para Equipe Médica

### Fatores de Risco Primários
1. **BMI elevado** - Preditor mais forte
2. **Falta de atividade física** - Aumenta significativamente obesidade
3. **Alto consumo de alimentos calóricos** - Forte associação
4. **Histórico familiar positivo** - Aumenta risco ~2x

### Fatores Protetores
1. **Atividade física regular** - 3-5x/semana
2. **Consumo adequado de vegetais** - Protetor importante
3. **Consumo de água elevado** - >2L/dia
4. **Monitoramento de calorias** - Indica consciência nutricional

### Protocolo Recomendado
1. Calcular BMI de todos os pacientes
2. Avaliar atividade física atual
3. Revisar padrão alimentar
4. Verificar consumo de água
5. Investigar histórico familiar
6. Implementar intervenção baseada na classificação

## 🔄 Pipeline de ML (Detalhado)

```
Dataset (2111 amostras)
        ↓
[Exploração e Limpeza]
        ↓
[Feature Engineering]
  - Criação de BMI
  - Features de interação
  - Encoding de categorias
        ↓
[Train-Test Split 80-20]
        ↓
    Treino (1688)    Teste (423)
        ↓                  ↓
[StandardScaler]          ↓
        ↓                  ↓
[Modelos Testados]        ↓
  - Ridge (94.80%)        ↓
  - RF (97.87%)           ↓
  - GB (98.35%) ✅        ↓
  - Voting (95.51%)       ↓
        ↓                  ↓
[Validação com Teste]
        ↓
    Acurácia: 98.35%
    F1-Score: 0.9834
    
[Deploy Streamlit]
```

## 🛠️ Tecnologias Utilizadas

- **Python 3.9+** - Linguagem principal
- **scikit-learn** - Modelagem ML
- **pandas/numpy** - Manipulação de dados
- **matplotlib/seaborn** - Visualizações
- **Streamlit** - Interface web interativa
- **joblib** - Serialização de modelos

## 📦 Requisitos do Projeto

- ✅ Pipeline de ML com feature engineering
- ✅ Modelo com >75% acurácia (98.35%)
- ✅ Deploy em Streamlit
- ✅ Análise e insights para equipe médica
- ✅ Código no GitHub
- ✅ Documentação completa

## 🔗 Links de Deploy (Placeholder)

*Após deploy, adicionar aqui:*
- 🌐 **App Streamlit**: [URL do app]
- 📊 **Dashboard Analítico**: [URL do painel]
- 💻 **Repositório GitHub**: [URL do repo]

## 📄 Arquivos de Entrega

- `01_ML_Pipeline_Obesity_Prediction.ipynb` - Notebook completo
- `02_Analise_Exploratoria_Insights.md` - Análise de insights
- `streamlit_app.py` - Aplicação Streamlit
- `README.md` - Este arquivo
- `models/` - Diretório com modelos treinados

## 👨‍💻 Autores

- **[Rafael Couto]** 
- **[Alex Oliveira]** 
- **[Ronaldo Rodrigues]** 

Desenvolvido como Tech Challenge 4 - Postech

## 📝 Licença

Este projeto é fornecido como é para fins educacionais.

---

**Última atualização**: 2024
**Status**: ✅ Pronto para produção
