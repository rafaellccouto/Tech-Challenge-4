import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Compatibilidade com diferentes versões do Streamlit
if hasattr(st, 'cache_resource'):
    cache_resource = st.cache_resource
elif hasattr(st, 'cache_data'):
    cache_resource = st.cache_data
else:
    cache_resource = st.cache

def render_divider():
    if hasattr(st, 'divider'):
        st.divider()
    else:
        st.markdown('---')

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Previsão de Obesidade",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CARREGAMENTO DE MODELOS E PREPROCESSADORES
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'

@cache_resource
def load_model():
    try:
        model = joblib.load(MODELS_DIR / 'best_obesity_model.pkl')
        scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
        label_encoder = joblib.load(MODELS_DIR / 'label_encoder.pkl')
        feature_names = joblib.load(MODELS_DIR / 'feature_names.pkl')
        metadata_path = MODELS_DIR / 'model_metadata.pkl'
        metadata = joblib.load(metadata_path) if metadata_path.exists() else None
        return model, scaler, label_encoder, feature_names, metadata
    except Exception as e:
        st.error("Erro ao carregar modelos salvos: " + str(e))
        st.info("Instale uma versão compatível do scikit-learn e reinicie o app.")
        return None, None, None, None, None

model, scaler, label_encoder, feature_names, metadata = load_model()

def preprocess_user_input(user_data_df, model_features):
    df = user_data_df.copy()
    
    # 1. Engenharia de Features (Sincronizada com o Notebook)
    caec_map = {'no': 4, 'Sometimes': 3, 'Frequently': 2, 'Always': 1}
    df['Snack_Health_Score'] = df['CAEC'].map(caec_map)
    df['Lifestyle_Balance'] = df['FAF'] - df['TUE']
    df['Water_per_Meal'] = df['CH2O'] / (df['NCP'] + 1)

    # 2. Encodings Binários
    df['family_history'] = (df['family_history'] == 'yes').astype(int)
    df['FAVC'] = (df['FAVC'] == 'yes').astype(int)
    df['SMOKE'] = (df['SMOKE'] == 'yes').astype(int)
    df['SCC'] = (df['SCC'] == 'yes').astype(int)
    df['Gender'] = (df['Gender'] == 'Male').astype(int)

    # 3. One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=['CAEC', 'CALC', 'MTRANS'])

    # 4. Alinhamento de Colunas (Garante que todas as colunas do treino existam)
    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Reordenar para a ordem exata das colunas do modelo
    return df_encoded[model_features]

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>Sistema de Previsão de Obesidade</h1>
        <p style='font-size: 18px; color: #666;'>
            Modelo de Machine Learning para auxiliar no diagnóstico de Obesidade
        </p>
    </div>
""", unsafe_allow_html=True)

render_divider()

# ============================================================================
# TABS PARA NAVEGAÇÃO
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Previsão", "📊 Análise de Dados", "ℹ️ Sobre o Modelo", "📖 Dicionário"]
)

# ============================================================================
# TAB 1: PREVISÃO
# ============================================================================
with tab1:
    st.header("Faça uma Previsão")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Dados Pessoais")
        sexo_pt = st.selectbox("Sexo", ["Feminino", "Masculino"], key="sexo")
        age = st.slider("Idade (anos)", 14, 61, 25)
        st.info("Este modelo realiza triagem usando apenas hábitos e histórico familiar. Altura, peso e IMC são usados apenas para validação clínica externa.")
        favc_pt = st.selectbox("Come alimentos altamente calóricos com frequência?", ["Não", "Sim"])

        vegetable_options = {1: "1 - Raramente", 2: "2 - Às vezes", 3: "3 - Sempre"}
        fcvc = st.selectbox("Frequência de consumo de vegetais", options=list(vegetable_options.keys()), format_func=lambda x: vegetable_options[x])
        ncp = st.selectbox("Número de refeições principais por dia", [1, 2, 3, 4])
        caec_pt = st.selectbox("Consome lanches entre refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
    
    with col2:
        st.subheader("🍽️ Hábitos Alimentares")
        family_history_pt = st.selectbox("Histórico familiar de obesidade?", ["Não", "Sim"])
        smoke_pt = st.selectbox("Fuma?", ["Não", "Sim"])

        water_options = {1: "1 - Menos de 1 litro por dia", 2: "2 - Entre 1 e 2 litros por dia", 3: "3 - Mais de 2 litros por dia"}
        ch2o = st.selectbox("Consumo diário de água (litros por dia)", options=list(water_options.keys()), format_func=lambda x: water_options[x])

        scc_pt = st.selectbox("Monitora as calorias que ingere?", ["Não", "Sim"])

        faf_options = {0: "0 - Nenhuma", 1: "1 - 1 a 2 vezes por semana", 2: "2 - 3 a 4 vezes por semana", 3: "3 - 5 ou mais vezes por semana"}
        faf = st.selectbox("Frequência de atividade física por semana", options=list(faf_options.keys()), format_func=lambda x: faf_options[x])
    
    with col3:
        st.subheader("📱 Estilo de Vida")
        screen_time_options = {0: "0 - Até 2 horas por dia", 1: "1 - Entre 3 e 5 horas por dia", 2: "2 - Mais de 5 horas por dia"}
        tue = st.selectbox("Tempo com dispositivos eletrônicos por dia", options=list(screen_time_options.keys()), format_func=lambda x: screen_time_options[x])
        calc_pt = st.selectbox("Consumo de bebida alcoólica", ["Não", "Às vezes", "Frequentemente", "Sempre"])
        mtrans_pt = st.selectbox("Meio de transporte habitual", ["Automóvel", "Moto", "Bicicleta", "Transporte Público", "Caminhada"])
    
    # Mapeamentos PT-BR -> Pipeline
    yesno_map = {"Não": "no", "Sim": "yes"}
    caec_map_ui = {"Não":"no", "Às vezes":"Sometimes", "Frequentemente":"Frequently", "Sempre":"Always"}
    mtrans_map = {"Automóvel":"Automobile", "Moto":"Motorbike", "Bicicleta":"Bike", "Transporte Público":"Public_Transportation", "Caminhada":"Walking"}

    if st.button("Fazer Previsão", use_container_width=True, type="primary"):
        if model is None or feature_names is None:
            st.error("Modelos não carregados. Verifique a pasta /models.")
            st.stop()

        # Criar DataFrame com os nomes de colunas originais do dataset (antes do preprocess_input)
        raw_data = pd.DataFrame({
            'Gender': ['Male' if sexo_pt == 'Masculino' else 'Female'],
            'Age': [float(age)],
            'family_history': [yesno_map[family_history_pt]],
            'FAVC': [yesno_map[favc_pt]],
            'FCVC': [float(fcvc)],
            'NCP': [float(ncp)],
            'CAEC': [caec_map_ui[caec_pt]],
            'SMOKE': [yesno_map[smoke_pt]],
            'CH2O': [float(ch2o)],
            'SCC': [yesno_map[scc_pt]],
            'FAF': [float(faf)],
            'TUE': [float(tue)],
            'CALC': [caec_map_ui[calc_pt]],
            'MTRANS': [mtrans_map[mtrans_pt]]
        })

        # Processamento e Predição
        try:
            X_processed = preprocess_user_input(raw_data, feature_names)
            X_scaled = scaler.transform(X_processed)
            pred_encoded = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0]
            pred_label = label_encoder.inverse_transform([pred_encoded])[0]

            class_name_map = {
                "Overweight_Level_I": "Sobrepeso Nível I", "Overweight_Level_II": "Sobrepeso Nível II",
                "Obesity_Type_I": "Obesidade Tipo I", "Obesity_Type_II": "Obesidade Tipo II",
                "Obesity_Type_III": "Obesidade Tipo III", "Normal_Weight": "Peso Normal",
                "Insufficient_Weight": "Peso Insuficiente"
            }
            pred_label_display = class_name_map.get(pred_label, pred_label)
            
            st.divider()
            col_pred1, col_pred2 = st.columns([2, 1])
            with col_pred1:
                st.markdown(f"<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; text-align: center; color: white;'><h2>Classificação Prevista</h2><h1 style='font-size: 48px;'>{pred_label_display}</h1><p>Confiança: <strong>{np.max(prob)*100:.1f}%</strong></p></div>", unsafe_allow_html=True)

            with col_pred2:
                st.markdown("### 📊 Probabilidades")
                classes = list(label_encoder.classes_)
                prob_df = pd.DataFrame({'Classe': [class_name_map.get(cl, cl) for cl in classes], 'Probabilidade': prob}).sort_values('Probabilidade', ascending=False)
                for _, row in prob_df.iterrows():
                    st.write(f"**{row['Classe']}**: {row['Probabilidade']*100:.1f}%")

            st.subheader("💡 Recomendações de Saúde")
            if faf < 1: st.info("🏃 Aumente atividade física para pelo menos 1x por semana")
            if ch2o < 2: st.info("💧 Aumente consumo de água para pelo menos 2L por dia")
            if favc_pt == "Sim": st.info("🍔 Reduza alimentos altamente calóricos")
        except Exception as e:
            st.error(f"Erro na predição: {e}")

# ============================================================================
# TAB 2: ANÁLISE DE DADOS
# ============================================================================
with tab2:
    st.header("📊 Análise de Dados e Insights")
    st.markdown("O modelo foi treinado com 2.111 amostras distribuídas em 7 classes de obesidade.")
    obesity_dist = {'Obesity_Type_I': 351, 'Obesity_Type_III': 324, 'Obesity_Type_II': 297, 'Overweight_Level_I': 290, 'Overweight_Level_II': 290, 'Normal_Weight': 287, 'Insufficient_Weight': 272}
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1: st.bar_chart(obesity_dist)
    with col_chart2: st.dataframe(pd.DataFrame(list(obesity_dist.items()), columns=['Classe', 'Frequência']), use_container_width=True)

# ============================================================================
# TAB 3: SOBRE O MODELO
# ============================================================================
with tab3:
    st.header("Informações Técnicas do Modelo")
    
    if metadata:
        # 1. Cards de Performance (Metricas em Percentual)
        st.subheader("Desempenho nos Testes")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("Acurácia Geral", f"{metadata['accuracy']*100:.1f}%")
        with col_m2:
            st.metric("Precisão Médio", f"{metadata['precision']*100:.1f}%")
        with col_m3:
            st.metric("Recall Médio", f"{metadata['recall']*100:.1f}%")
        with col_m4:
            st.metric("F1-Score", f"{metadata['f1_score']*100:.1f}%")
            
        st.divider()
        
        # 2. Detalhes da Arquitetura
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.subheader("Algoritmo Selecionado")
            st.info(f"**Tipo:** {metadata['model_name']}")
            st.write("Este modelo foi selecionado por apresentar o melhor equilíbrio entre sensibilidade e precisão para detectar diferentes níveis de obesidade.")
            
        with col_info2:
            st.subheader("Variáveis de Entrada (Features)")
            # Exibir features em uma lista organizada em colunas menores
            feats = metadata['feature_names']
            # Dividindo a lista de features para nao ficar uma coluna muito longa
            half = len(feats) // 2
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                for f in feats[:half]: st.markdown(f"- {f}")
            with sub_col2:
                for f in feats[half:]: st.markdown(f"- {f}")
    else:
        st.warning("Metadados não encontrados. Certifique-se de que o arquivo 'model_metadata.pkl' está na pasta /models.")

# ============================================================================
# TAB 4: DICIONÁRIO
# ============================================================================
with tab4:
    st.header("📖 Dicionário de Dados")
    data_dict = {"Sexo": "Feminino/Masculino", "Age": "Idade em anos", "family_history": "Histórico familiar de excesso de peso", "FAVC": "Consumo frequente de alimentos calóricos", "FCVC": "Consumo de vegetais (1-3)", "FAF": "Atividade física semanal", "TUE": "Tempo de tela diário"}
    st.dataframe(pd.DataFrame(list(data_dict.items()), columns=['Variável', 'Descrição']), use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
render_divider()
st.markdown("<div style='text-align: center; color: #999; padding: 20px;'><p>Sistema de Previsão de Obesidade | Postech - Tech Challenge 4</p></div>", unsafe_allow_html=True)