# 🏥 Sistema de Previsão de Obesidade - Tech Challenge Postech

## 📋 Descrição do Projeto

Sistema de Machine Learning para auxiliar médicos e médicas na previsão e diagnóstico de obesity. O modelo utiliza dados de hábitos, histórico familiar, alimentação e estilo de vida para classificar o nível de obesidade em 7 categorias. O diferencial desta solução é a operação **sem dados antropométricos (Peso/Altura)**, focando na triagem comportamental.

**Acurácia: 83.22%** ✅ (Acima do target de 75%)

## 🎯 Objetivos Alcançados

-  **Pipeline completo de ML** com feature engineering e treinamento focado em 27 variáveis comportamentais.
-  **Modelo com 83.22% de acurácia** (Random Forest).
-  **Deploy em Streamlit** - Aplicação interativa com níveis de confiança.
-  **Análise exploratória** com insights para equipe médica.
-  **Código documentado** e compartilhado no GitHub.

## 📊 Estrutura do Projeto

```
Tech Challenge 4/
├── 01_ML_Pipeline_Obesity_Prediction.ipynb  # Notebook com pipeline completo
├── streamlit_app.py                          # Aplicação Streamlit
├── models/                                   # Modelos treinados
│   ├── best_obesity_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
├── Obesity.csv                               # Dataset original
└── README.md                                 # Este arquivo
```

## 🚀 Como Executar

### Pré-requisitos
- Python 3.9+
- pip

### 1. Instalar Dependências
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### 2. Executar a Aplicação Streamlit
```bash
streamlit run streamlit_app.py
```

## 📈 Resultados do Modelo

### Performance Geral

| Métrica | Valor |
|---------|-------|
| **Acurácia** | **83.22%** |
| Precisão (weighted) | 84.14% |
| Recall (weighted) | 83.22% |
| F1-Score (weighted) | 83.38% |

### Modelo Selecionado
- **Random Forest** selecionado por atingir 83.22% de acurácia no pipeline sem antropometria, mantendo estabilidade entre as classes.

### Distribuição de Classes
Dataset contém 2.111 amostras distribuídas em 7 classes, incluindo: Obesity Type I (16.6%), Obesity Type III (15.3%), Obesity Type II (14.1%), entre outras.

## 🔍 Dicionário de Dados (Features Principais)

| Variável | Descrição | Valores |
|----------|-----------|---------|
| Gender | Gênero | Female, Male |
| Age | Idade | 14-61 anos |
| family_history | Histórico familiar | yes, no |
| FCVC | Frequência de vegetais | 1-3 |
| FAF | Atividade física/semana | 0-3 |
| TUE | Tempo com eletrônicos | 0-2 |
| Lifestyle_Balance | Atividade vs Sedentarismo | Calculada (FAF - TUE) |

## 💡 Features Mais Importantes

As variáveis com maior impacto na previsão segundo o Random Forest:
1. **FCVC** (Consumo de vegetais)
2. **Age** (Idade)
3. **Gender** (Gênero)
4. **NCP** (Número de refeições)
5. **Water_per_Meal** (Hidratação relativa)

## 🔬 Metodologia

### Fases da Pipeline
1. **EDA**: Análise de 2.111 amostras e distribuições.
2. **Preparação**: Tratamento decimal e encoding (Label/One-Hot).
3. **Feature Engineering**: Criação de `Snack_Health_Score`, `Lifestyle_Balance` e `Water_per_Meal`.
4. **Modelagem**: Split 80/20 estratificado e StandardScaler (Anti-Leakage).
5. **Validação**: Cross-Validation (5-fold) e análise de Overfitting (Gap de 12.99% no RF).

## 🎨 Interface da Aplicação Streamlit
A aplicação oferece abas de **Previsão** (com cálculo de probabilidade e nível de confiança), **Análise de Dados** e **Sobre o Modelo**.

## 📚 Insights para Equipe Médica
- **Consumo de Vegetais (FCVC)**: Principal preditor para distinguir peso normal de sobrepeso inicial.
- **Janela de Idade**: Risco crítico detectado na transição entre 18-25 anos.
- **Histórico Familiar**: Atua como fator multiplicador de risco nas classes de Obesidade Tipo II e III.

## 🔗 Links de Acesso
- 🌐 **App Streamlit**: [https://tech-challenge-4-previsao-obesidade.streamlit.app/](https://tech-challenge-4-previsao-obesidade.streamlit.app/)
- 💻 **Repositório GitHub**: [https://github.com/rafaellccouto/Tech-Challenge-4](https://github.com/rafaellccouto/Tech-Challenge-4)

## 👨‍💻 Autores
- **Rafael Couto** 
- **Alex Oliveira** 
- **Ronaldo Rodrigues** 

*Desenvolvido como Tech Challenge 4 - Postech.2026*