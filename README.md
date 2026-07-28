# 🏥 Sistema de Previsão de Obesidade - Tech Challenge Postech

## 📋 Descrição do Projeto

Sistema de Machine Learning para auxiliar médicos e médicas na previsão e diagnóstico de obesity. O modelo utiliza dados de hábitos, histórico familiar, alimentação e estilo de vida para classificar o nível de obesidade em 7 categorias.

**Acurácia: 86.29%** ✅ (Acima do target de 75%)

## 🎯 Objetivos Alcançados

- ✅ **Pipeline completo de ML** com feature engineering e treinamento
- ✅ **Modelo com 86.29% de acurácia** (Random Forest)
- ✅ **Deploy em Streamlit** - Aplicação interativa
- ✅ **Análise exploratória** com insights para equipe médica
- ✅ **Código documentado** e compartilhado no GitHub

## 📊 Estrutura do Projeto

```
Tech Challenge 4/
├── 01_ML_Pipeline_Obesity_Prediction.ipynb  # Notebook com pipeline completo
├── 02_Analise_Exploratoria_Insights.md      # Insights analíticos
├── streamlit_app.py                          # Aplicação Streamlit
├── train_no_anthropometry.py                 # Script de treino sem antropometria
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

> Recomendado: use Python 3.11 para compatibilidade com a versão de scikit-learn usada no deploy.

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
| **Acurácia** | **86.29%** |
| Precisão (média) | 86.10% |
| Recall (média) | 86.29% |
| F1-Score (média) | 86.15% |

### Modelo Selecionado

- **Random Forest** com melhor desempenho no pipeline sem antropometria

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

As variáveis com maior impacto na previsão (segundo Random Forest) no modelo sem antropometria:

1. **Gender** (~25%) - Gênero
2. **Age_Family_History** (~20%) - Interação Idade × Histórico Familiar
3. **family_history** (~15%) - Histórico familiar de obesidade
4. **FAVC** (~12%) - Consumo de alimentos calóricos
5. **FAF** (~10%) - Frequência de atividade física
6. **CH2O** (~8%) - Consumo diário de água
7. **FCVC** (~6%) - Consumo de vegetais

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
  - Age_Family_History = Age × family_history
  - One-hot encoding de CAEC, CALC e MTRANS

#### 3. Modelagem
- Train-Test Split: 80% treino, 20% teste (estratificado)
- Normalização: StandardScaler
- Validação: 5-fold Cross-Validation
- Hiperparametrização: GridSearchCV

#### 5. Modelos Testados
- Ridge Logistic Regression (L2 Regularization)
- Random Forest (200 estimadores)
- Gradient Boosting (testado para comparação)
- Voting Classifier (ensemble)

#### 6. Seleção do Modelo
**Random Forest** selecionado por melhor desempenho e robustez no pipeline atual.

## 🎨 Interface da Aplicação Streamlit

A aplicação oferece 4 abas principais:

### 1. 🔮 Previsão
- Formulário interativo para entrada de hábitos e histórico familiar
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
1. **Histórico familiar positivo** - Preditor importante para triagem
2. **Falta de atividade física** - Aumenta significativamente o risco de obesidade
3. **Alto consumo de alimentos calóricos** - Forte associação com sobrepeso
4. **Baixo consumo de água** - Reforça desequilíbrio metabólico

### Fatores Protetores
1. **Atividade física regular** - 3-5x/semana
2. **Consumo adequado de vegetais** - Protetor importante
3. **Consumo de água elevado** - >2L/dia
4. **Monitoramento de calorias** - Indica consciência nutricional

### Protocolo Recomendado
1. Avaliar hábitos e histórico familiar para triagem inicial
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
  - Ridge Logistic Regression
  - Random Forest ✅
  - Gradient Boosting
  - Voting Classifier
        ↓                  ↓
[Validação com Teste]
        ↓
    Acurácia: 86.29%
    F1-Score: 0.8615
    
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
- ✅ Modelo com >75% acurácia (86.29%)
- ✅ Deploy em Streamlit
- ✅ Análise e insights para equipe médica
- ✅ Código no GitHub
- ✅ Documentação completa

## 🔗 Links de Deploy (Placeholder)

*Após deploy, adicionar aqui:*
- 🌐 **App Streamlit**: https://tech-challenge-4-previsao-obesidade.streamlit.app/
- 📊 **Dashboard Analítico**: [URL do painel]
- 💻 **Repositório GitHub**: https://github.com/rafaellccouto/Tech-Challenge-4

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
