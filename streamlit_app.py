import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="🏥 Previsão de Obesidade",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CARREGAMENTO DE MODELOS E PREPROCESSADORES
# ============================================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_obesity_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, scaler, label_encoder, feature_names
    except Exception as e:
        st.error("Erro ao carregar modelos: " + str(e))
        return None, None, None, None

model, scaler, label_encoder, feature_names = load_model()

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🏥 Sistema de Previsão de Obesidade</h1>
        <p style='font-size: 18px; color: #666;'>
            Modelo de Machine Learning para auxiliar no diagnóstico de Obesidade
        </p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# ============================================================================
# TABS PARA NAVEGAÇÃO
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Previsão", "📊 Análise de Dados", "ℹ️ Sobre o Modelo", "📖 Dicionário"]
)

# ============================================================================
# TAB 1: PREVISÃO
# ============================================================================
with tab1:
    st.header("🔮 Faça uma Previsão")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Dados Pessoais")
        
        sexo = st.selectbox("Sexo", ["Feminino", "Masculino"], key="sexo")
        age = st.slider("Idade (anos)", 14, 61, 25)
        height = st.slider("Altura (metros)", 1.45, 1.98, 1.70, step=0.01)
        weight = st.slider("Peso (kg)", 39.0, 173.0, 75.0, step=0.5)
        
    with col2:
        st.subheader("🍽️ Hábitos Alimentares")
        
        family_history = st.selectbox("Histórico familiar de obesidade?", ["no", "yes"])
        favc = st.selectbox("Come alimentos altamente calóricos com frequência?", ["no", "yes"])
        fcvc = st.selectbox("Frequência de consumo de vegetais", [1, 2, 3], 
                           help="1: Raramente, 2: Às vezes, 3: Sempre")
        ncp = st.selectbox("Número de refeições principais por dia", [1, 2, 3, 4])
        caec = st.selectbox("Consome lanches entre refeições?", 
                           ["no", "Sometimes", "Frequently", "Always"])
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🚴 Atividades e Hábitos")
        
        smoke = st.selectbox("Fuma?", ["no", "yes"])
        ch2o = st.selectbox("Consumo de água diário", [1, 2, 3],
                           help="1: <1L/dia, 2: 1-2L/dia, 3: >2L/dia")
        scc = st.selectbox("Monitora as calorias que ingere?", ["no", "yes"])
        faf = st.selectbox("Frequência de atividade física por semana", [0, 1, 2, 3],
                          help="0: Nenhuma, 1: 1-2x, 2: 3-4x, 3: 5x ou mais")
    
    with col4:
        st.subheader("📱 Estilo de Vida")
        
        tue = st.selectbox("Tempo com dispositivos eletrônicos/dia", [0, 1, 2],
                          help="0: 0-2h, 1: 3-5h, 2: >5h")
        calc = st.selectbox("Consumo de bebida alcoólica", 
                           ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Meio de transporte habitual",
                             ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])
    
    # ========================================================================
    # PROCESSAMENTO E PREVISÃO
    # ========================================================================
    if st.button("🔮 Fazer Previsão", use_container_width=True, type="primary"):

        # Verifica se os modelos foram carregados corretamente
        if model is None or scaler is None or label_encoder is None or feature_names is None:
            st.error("Modelos não carregados. Verifique os arquivos em /models e os logs de carregamento.")
            st.stop()

        # Preparar dados (usar nomes compatíveis com feature_names do modelo)
        input_data = {
            'Gender': 1 if sexo == 'Masculino' else 0,
            'Age': float(age),
            'Height': float(height),
            'Weight': float(weight),
            'family_history': 1 if family_history == 'yes' else 0,
            'FAVC': 1 if favc == 'yes' else 0,
            'FCVC': int(fcvc),
            'NCP': int(ncp),
            'SMOKE': 1 if smoke == 'yes' else 0,
            'CH2O': int(ch2o),
            'SCC': 1 if scc == 'yes' else 0,
            'FAF': int(faf),
            'TUE': int(tue),
            'MTRANS_Automobile': 1 if mtrans == 'Automobile' else 0,
            'MTRANS_Bike': 1 if mtrans == 'Bike' else 0,
            'MTRANS_Motorbike': 1 if mtrans == 'Motorbike' else 0,
            'MTRANS_Public_Transportation': 1 if mtrans == 'Public_Transportation' else 0,
            'MTRANS_Walking': 1 if mtrans == 'Walking' else 0,
            'CAEC_Always': 1 if caec == 'Always' else 0,
            'CAEC_Frequently': 1 if caec == 'Frequently' else 0,
            'CAEC_Sometimes': 1 if caec == 'Sometimes' else 0,
            'CAEC_no': 1 if caec == 'no' else 0,
            'CALC_Always': 1 if calc == 'Always' else 0,
            'CALC_Frequently': 1 if calc == 'Frequently' else 0,
            'CALC_Sometimes': 1 if calc == 'Sometimes' else 0,
            'CALC_no': 1 if calc == 'no' else 0,
        }

        # Calcular features de engenharia
        bmi = float(weight) / (float(height) ** 2)
        faf_weight = int(faf) * float(weight)
        age_family = float(age) * (1 if family_history == 'yes' else 0)
        weight_height_ratio = float(weight) / float(height)

        input_data['BMI'] = bmi
        input_data['FAF_Weight'] = faf_weight
        input_data['Age_Family_History'] = age_family
        input_data['Weight_Height_Ratio'] = weight_height_ratio

        # Criar DataFrame e validar colunas
        df_input = pd.DataFrame([input_data])

        # Verifica se feature_names é iterável e contém strings
        try:
            expected_cols = list(feature_names)
        except Exception:
            st.error("feature_names inválido. Verifique o arquivo models/feature_names.pkl")
            st.stop()

        missing = [c for c in expected_cols if c not in df_input.columns]
        if missing:
            st.error(f"Faltam colunas no input: {missing}")
            st.write("Colunas disponíveis no input:", df_input.columns.tolist())
            st.stop()
        else:
            df_input = df_input[expected_cols]

        # Normalizar e prever (captura de exceções)
        try:
            X_scaled = scaler.transform(df_input)
            pred_encoded = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0]
        except Exception as e:
            st.error("Erro durante a predição: " + str(e))
            st.stop()

        # Decodificar e mostrar resultado
        try:
            pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        except Exception:
            pred_label = str(pred_encoded)

        conf = float(np.max(prob) * 100)

        st.divider()

        col_pred1, col_pred2 = st.columns([2, 1])

        with col_pred1:
            st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 30px;
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                '>
                    <h2>Classificação Prevista</h2>
                    <h1 style='font-size: 48px; margin: 20px 0;'>{pred_label}</h1>
                    <p style='font-size: 20px;'>Confiança: <strong>{conf:.1f}%</strong></p>
                </div>
            """, unsafe_allow_html=True)

        with col_pred2:
            st.markdown("### 📊 Probabilidades")
            try:
                classes = list(label_encoder.classes_)
            except Exception:
                classes = [str(i) for i in range(len(prob))]
            prob_df = pd.DataFrame({
                'Classe': classes,
                'Probabilidade': prob
            }).sort_values('Probabilidade', ascending=False)

            for idx, row in prob_df.iterrows():
                st.write(f"**{row['Classe']}**: {row['Probabilidade']*100:.1f}%")

        st.divider()

        # Dicas de saúde (mantém seu código)
        st.subheader("💡 Recomendações de Saúde")
        recommendations = []
        if bmi < 18.5:
            recommendations.append("⚠️ **Peso baixo**: Consulte um médico para avaliar deficiência nutricional")
        elif bmi < 25:
            recommendations.append("✅ **Peso normal**: Mantenha os hábitos saudáveis!")
        elif bmi < 30:
            recommendations.append("⚠️ **Sobrepeso**: Aumente atividade física e revise a alimentação")
        else:
            recommendations.append("🚨 **Obesidade**: Busque orientação médica profissional urgentemente")

        if int(faf) < 1:
            recommendations.append("🏃 Aumente atividade física para pelo menos 1x por semana")
        if int(ch2o) < 2:
            recommendations.append("💧 Aumente consumo de água para pelo menos 2L por dia")
        if favc == "yes":
            recommendations.append("🍔 Reduza alimentos altamente calóricos")
        if int(fcvc) < 3:
            recommendations.append("🥗 Aumente consumo de vegetais nas refeições")

        for rec in recommendations:
            st.info(rec)

# ============================================================================
# TAB 2: ANÁLISE DE DADOS
# ============================================================================
with tab2:
    st.header("📊 Análise de Dados e Insights")
    
    st.markdown("""
    ### Distribuição de Classes no Dataset de Treinamento
    
    O modelo foi treinado com 2.111 amostras distribuídas em 7 classes de obesidade:
    """)
    
    obesity_dist = {
        'Obesity_Type_I': 351,
        'Obesity_Type_III': 324,
        'Obesity_Type_II': 297,
        'Overweight_Level_I': 290,
        'Overweight_Level_II': 290,
        'Normal_Weight': 287,
        'Insufficient_Weight': 272
    }
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.bar_chart(obesity_dist)
    
    with col_chart2:
        df_dist = pd.DataFrame(list(obesity_dist.items()), columns=['Classe', 'Frequência'])
        df_dist['Proporção %'] = (df_dist['Frequência'] / df_dist['Frequência'].sum() * 100).round(1)
        st.dataframe(df_dist, use_container_width=True)
    
    st.divider()
    
    st.markdown("""
    ### Features Mais Importantes
    
    As seguintes variáveis têm maior impacto na previsão:
    
    1. **BMI** (Índice de Massa Corporal) - ~30%
    2. **Weight_Height_Ratio** (Proporção Peso/Altura) - ~15%
    3. **Weight** (Peso) - ~10%
    4. **Sexo** (Sexo) - ~8%
    5. **Age_Family_History** (Idade × Histórico Familiar) - ~6%
    """)
    
    st.divider()
    
    st.markdown("""
    ### Variáveis Utilizadas no Modelo
    
    - **Antropométricas**: Idade, Altura, Peso, BMI, Proporção Peso/Altura
    - **Alimentares**: FAVC, FCVC, NCP, CAEC
    - **Hábitos**: SMOKE, CH2O, SCC, FAF, TUE, CALC
    - **Transporte**: MTRANS
    - **Genética**: family_history
    """)

# ============================================================================
# TAB 3: SOBRE O MODELO
# ============================================================================
with tab3:
    st.header("ℹ️ Sobre o Modelo")
    
    col_model1, col_model2 = st.columns(2)
    
    with col_model1:
        st.markdown("""
        ### 🤖 Algoritmo Utilizado
        
        **Gradient Boosting Classifier**
        
        - N-estimadores: 200
        - Learning rate: 0.1
        - Profundidade máxima: 5
        - Framework: scikit-learn
        """)
        
        st.markdown("""
        ### 📈 Desempenho
        
        | Métrica | Valor |
        |---------|-------|
        | Acurácia | 98.35% |
        | Precisão (média) | 98.37% |
        | Recall (média) | 98.35% |
        | F1-Score (média) | 98.34% |
        """)
    
    with col_model2:
        st.markdown("""
        ### 📊 Modelos Testados
        
        - Ridge Logistic: 94.80%
        - Random Forest: 97.87%
        - **Gradient Boosting: 98.35%** ✅
        - Voting Ensemble: 95.51%
        
        **Melhor modelo selecionado**: Gradient Boosting
        """)
        
        st.markdown("""
        ### ✅ Requisitos Atendidos
        
        - ✅ Acurácia > 75% (98.35%)
        - ✅ Pipeline completo de ML
        - ✅ Feature engineering
        - ✅ Deploy em Streamlit
        """)
    
    st.divider()
    
    st.markdown("""
    ### 🔬 Metodologia
    
    **1. Exploração de Dados (EDA)**
    - Análise de distribuição de classes
    - Verificação de dados faltantes
    - Análise estatística descritiva
    
    **2. Preparação de Dados**
    - Limpeza e arredondamento de variáveis categóricas
    - Encoding: Label encoding para variáveis binárias, One-Hot para multicategoriais
    - Feature engineering: Criação de BMI, interações, proporções
    
    **3. Modelagem**
    - Train-test split estratificado (80-20)
    - Normalização com StandardScaler
    - GridSearchCV para otimização de hiperparâmetros
    - Cross-validation estratificada (5-fold)
    
    **4. Validação**
    - Avaliação em conjunto de teste
    - Matriz de confusão
    - Relatório de classificação por classe
    """)

# ============================================================================
# TAB 4: DICIONÁRIO DE DADOS
# ============================================================================
with tab4:
    st.header("📖 Dicionário de Dados")
    
    data_dict = {
        "Sexo": "Sexo (Feminino/Masculino)",
        "Age": "Idade em anos (14-61)",
        "Height": "Altura em metros (1.45-1.98)",
        "Weight": "Peso em kg (39-173)",
        "family_history": "Histórico familiar de excesso de peso (yes/no)",
        "FAVC": "Consumo frequente de alimentos muito calóricos (yes/no)",
        "FCVC": "Frequência de consumo de vegetais (1=Raramente, 2=Às vezes, 3=Sempre)",
        "NCP": "Número de refeições principais/dia (1-4)",
        "CAEC": "Consome lanches entre refeições (no/Sometimes/Frequently/Always)",
        "SMOKE": "Hábito de fumar (yes/no)",
        "CH2O": "Consumo diário de água (1=<1L, 2=1-2L, 3=>2L)",
        "SCC": "Monitora ingestão calórica diária (yes/no)",
        "FAF": "Frequência semanal de atividade física (0=Nenhuma, 1=1-2x, 2=3-4x, 3=5x+)",
        "TUE": "Tempo com dispositivos eletrônicos (0=0-2h, 1=3-5h, 2=>5h)",
        "CALC": "Consumo de bebida alcoólica (no/Sometimes/Frequently/Always)",
        "MTRANS": "Meio de transporte habitual",
        "Obesity": "Nível de obesidade (classe alvo)"
    }
    
    df_dict = pd.DataFrame(list(data_dict.items()), columns=['Variável', 'Descrição'])
    st.dataframe(df_dict, use_container_width=True)
    
    st.divider()
    
    st.markdown("""
    ### Classificações de Obesidade
    
    | Classe | Descrição |
    |--------|-----------|
    | Insufficient_Weight | Abaixo do peso |
    | Normal_Weight | Peso normal |
    | Overweight_Level_I | Sobrepeso Nível I |
    | Overweight_Level_II | Sobrepeso Nível II |
    | Obesity_Type_I | Obesidade Tipo I |
    | Obesity_Type_II | Obesidade Tipo II |
    | Obesity_Type_III | Obesidade Tipo III (mórbida) |
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
    <div style='text-align: center; color: #999; padding: 20px;'>
        <p>🏥 Sistema de Previsão de Obesidade | Postech - Tech Challenge 4</p>
        <p>Desenvolvido com Python, Streamlit e Machine Learning</p>
        <p>Modelo: Gradient Boosting | Acurácia: 98.35%</p>
    </div>
""", unsafe_allow_html=True)
