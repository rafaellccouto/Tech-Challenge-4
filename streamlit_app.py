import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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


@st.cache_resource
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

# Mostrar feature_names em um expander para debug (remova em produção)
# with st.expander("Debug: nomes de features (apenas para desenvolvimento)", expanded=False):
#     st.write("feature_names:", feature_names)

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

st.divider()

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

        vegetable_options = {
            1: "1 - Raramente",
            2: "2 - Às vezes",
            3: "3 - Sempre"
        }
        fcvc = st.selectbox(
            "Frequência de consumo de vegetais",
            options=list(vegetable_options.keys()),
            format_func=lambda x: vegetable_options[x],
            help="Selecione com que frequência você costuma comer vegetais nas refeições."
        )
        ncp = st.selectbox("Número de refeições principais por dia", [1, 2, 3, 4])
        caec_pt = st.selectbox("Consome lanches entre refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
    
    with col2:
        st.subheader("🍽️ Hábitos Alimentares")
        family_history_pt = st.selectbox("Histórico familiar de obesidade?", ["Não", "Sim"])
        smoke_pt = st.selectbox("Fuma?", ["Não", "Sim"])

        water_options = {
            1: "1 - Menos de 1 litro por dia",
            2: "2 - Entre 1 e 2 litros por dia",
            3: "3 - Mais de 2 litros por dia"
        }
        ch2o = st.selectbox(
            "Consumo diário de água (litros por dia)",
            options=list(water_options.keys()),
            format_func=lambda x: water_options[x],
            help="Selecione a faixa aproximada de litros de água que você bebe por dia."
        )

        scc_pt = st.selectbox("Monitora as calorias que ingere?", ["Não", "Sim"])

        faf_options = {
            0: "0 - Nenhuma",
            1: "1 - 1 a 2 vezes por semana",
            2: "2 - 3 a 4 vezes por semana",
            3: "3 - 5 ou mais vezes por semana"
        }
        faf = st.selectbox(
            "Frequência de atividade física por semana (número de vezes)",
            options=list(faf_options.keys()),
            format_func=lambda x: faf_options[x],
            help="Escolha quantas vezes por semana você pratica atividade física."
        )
    
    with col3:
        st.subheader("📱 Estilo de Vida")

        screen_time_options = {
            0: "0 - Até 2 horas por dia",
            1: "1 - Entre 3 e 5 horas por dia",
            2: "2 - Mais de 5 horas por dia"
        }
        tue = st.selectbox(
            "Tempo com dispositivos eletrônicos por dia (horas)",
            options=list(screen_time_options.keys()),
            format_func=lambda x: screen_time_options[x],
            help="Selecione a faixa de horas diárias que você costuma usar dispositivos eletrônicos."
        )
        calc_pt = st.selectbox("Consumo de bebida alcoólica", ["Não", "Às vezes", "Frequentemente", "Sempre"])
        mtrans_pt = st.selectbox("Meio de transporte habitual",
                                 ["Automóvel", "Moto", "Bicicleta", "Transporte Público", "Caminhada"])
    
    # Mapeamentos PT-BR -> valores/nomes esperados pelo pipeline
    sexo_map = {"Feminino": 0, "Masculino": 1}
    yesno_map = {"Não": "no", "Sim": "yes"}
    caec_map = {"Não":"no","Às vezes":"Sometimes","Frequentemente":"Frequently","Sempre":"Always"}
    calc_map = {"Não":"no","Às vezes":"Sometimes","Frequentemente":"Frequently","Sempre":"Always"}
    mtrans_map = {
        "Automóvel":"Automobile",
        "Moto":"Motorbike",
        "Bicicleta":"Bike",
        "Transporte Público":"Public_Transportation",
        "Caminhada":"Walking"
    }

    # ========================================================================
    # PROCESSAMENTO E PREVISÃO
    # ========================================================================
    if st.button("Fazer Previsão", use_container_width=True, type="primary"):

        # Verifica se os modelos foram carregados corretamente
        if model is None or scaler is None or label_encoder is None or feature_names is None:
            st.error("Modelos não carregados. Verifique os arquivos em /models e os logs de carregamento.")
            st.stop()

        # Normaliza nomes de feature_names (remove espaços acidentais)
        try:
            expected_cols = [str(c).strip() for c in list(feature_names)]
        except Exception:
            st.error("feature_names inválido. Verifique o arquivo models/feature_names.pkl")
            st.stop()

        # Mapeia valores da UI para os nomes/valores que o modelo espera
        family_history = yesno_map[family_history_pt]
        favc = yesno_map[favc_pt]
        smoke = yesno_map[smoke_pt]
        scc = yesno_map[scc_pt]
        caec = caec_map[caec_pt]
        calc = calc_map[calc_pt]
        mtrans = mtrans_map[mtrans_pt]

        # Monta input_data com nomes compatíveis (ajusta dinamicamente o nome da coluna de gênero)
        gender_col = 'Gender' if 'Gender' in expected_cols else ('gender' if 'gender' in expected_cols else 'Gender')
        input_data = {
            gender_col: sexo_map[sexo_pt],
            'Age': float(age),
            'family_history': 1 if family_history == 'yes' else 0,
            'FAVC': 1 if favc == 'yes' else 0,
            'FCVC': int(fcvc),
            'NCP': int(ncp),
            'SMOKE': 1 if smoke == 'yes' else 0,
            'CH2O': int(ch2o),
            'SCC': 1 if scc == 'yes' else 0,
            'FAF': int(faf),
            'TUE': int(tue),
            # MTRANS dummies (cria as colunas esperadas pelo modelo)
            'MTRANS_Automobile': 1 if mtrans == 'Automobile' else 0,
            'MTRANS_Bike': 1 if mtrans == 'Bike' else 0,
            'MTRANS_Motorbike': 1 if mtrans == 'Motorbike' else 0,
            'MTRANS_Public_Transportation': 1 if mtrans == 'Public_Transportation' else 0,
            'MTRANS_Walking': 1 if mtrans == 'Walking' else 0,
            # CAEC dummies
            'CAEC_Always': 1 if caec == 'Always' else 0,
            'CAEC_Frequently': 1 if caec == 'Frequently' else 0,
            'CAEC_Sometimes': 1 if caec == 'Sometimes' else 0,
            'CAEC_no': 1 if caec == 'no' else 0,
            # CALC dummies
            'CALC_Always': 1 if calc == 'Always' else 0,
            'CALC_Frequently': 1 if calc == 'Frequently' else 0,
            'CALC_Sometimes': 1 if calc == 'Sometimes' else 0,
            'CALC_no': 1 if calc == 'no' else 0,
        }

        # Criar DataFrame e validar colunas
        df_input = pd.DataFrame([input_data])
        # Normaliza nomes de colunas do df_input
        df_input.columns = [str(c).strip() for c in df_input.columns]

        missing = [c for c in expected_cols if c not in df_input.columns]
        extra = [c for c in df_input.columns if c not in expected_cols]

        if missing:
            st.error("Faltam colunas no input: " + ", ".join(missing))
            with st.expander("Ver detalhes do input e expected_cols"):
                st.write("expected_cols:", expected_cols)
                st.write("df_input.columns:", df_input.columns.tolist())
                st.write("df_input row:", df_input.iloc[0].to_dict())
            st.stop()

        # Reordena colunas conforme esperado
        df_input = df_input[expected_cols]
        

        # Normalizar e prever (captura de exceções)
        try:
            X_raw = df_input.values
            X_scaled = scaler.transform(X_raw)
            pred_encoded = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0].astype(float)

            # Garantir que as probabilidades somem a 1 (normalização de segurança)
            prob = np.clip(prob, 0.0, None)
            prob_sum = float(np.sum(prob))
            if prob_sum <= 0 or np.isnan(prob_sum):
                raise ValueError("Probabilidades inválidas retornadas pelo modelo")
            if not np.isclose(prob_sum, 1.0, atol=1e-6):
                prob = prob / prob_sum
        except Exception as e:
            st.error("Erro durante a predição: " + str(e))
            st.stop()

        # Decodificar e mostrar resultado
        try:
            pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        except Exception:
            pred_label = str(pred_encoded)

        class_name_map = {
            "Overweight_Level_I": "Sobrepeso Nível I",
            "Overweight_Level_II": "Sobrepeso Nível II",
            "Obesity_Type_I": "Obesidade Tipo I",
            "Obesity_Type_II": "Obesidade Tipo II",
            "Obesity_Type_III": "Obesidade Tipo III",
            "Normal_Weight": "Peso Normal",
            "Insufficient_Weight": "Peso Insuficiente"
        }
        pred_label_display = class_name_map.get(pred_label, pred_label)

        conf = float(np.max(prob) * 100)

        # Expander com debug da predição (model.classes_, label_encoder, prob, df_input)
        # with st.expander("🔍 Debug da predição (apenas dev)", expanded=False):
        #     st.write("Número de features do scaler:", scaler.mean_.shape[0])
        #     st.write("expected_cols:", expected_cols)
        #     st.write("df_input (colunas):", df_input.columns.tolist())
        #     st.write("df_input (valores):", df_input.iloc[0].to_dict())
        #     st.write("model.classes_ (codificadas):", getattr(model, "classes_", None))
        #     try:
        #         st.write("label_encoder.classes_ (nomes):", list(label_encoder.classes_))
        #     except Exception:
        #         st.write("label_encoder sem classes_")
        #     st.write("prob array:", prob.tolist())

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
                    <h1 style='font-size: 48px; margin: 20px 0;'>{pred_label_display}</h1>
                    <p style='font-size: 20px;'>Confiança: <strong>{conf:.1f}%</strong></p>
                </div>
            """, unsafe_allow_html=True)

        with col_pred2:
            st.markdown("### 📊 Probabilidades")
            # --- Alinha probabilidades com os rótulos corretos ---
            try:
                encoded_model_classes = getattr(model, "classes_", None)
                if encoded_model_classes is not None:
                    classes = list(label_encoder.inverse_transform(encoded_model_classes))
                else:
                    classes = list(label_encoder.classes_)
            except Exception:
                classes = [str(i) for i in range(len(prob))]

            classes_display = [class_name_map.get(cl, cl) for cl in classes]

            prob_df = pd.DataFrame({
                'Classe': classes_display,
                'Probabilidade': prob
            }).sort_values('Probabilidade', ascending=False)

            for idx, row in prob_df.iterrows():
                st.write(f"**{row['Classe']}**: {row['Probabilidade']*100:.1f}%")

        st.divider()

        # Recomendações de saúde
        st.subheader("💡 Recomendações de Saúde")
        recommendations = []
        if int(faf) < 1:
            recommendations.append("🏃 Aumente atividade física para pelo menos 1x por semana")
        if int(ch2o) < 2:
            recommendations.append("💧 Aumente consumo de água para pelo menos 2L por dia")
        if favc == "yes":
            recommendations.append("🍔 Reduza alimentos altamente calóricos")
        if int(fcvc) < 3:
            recommendations.append("🥗 Aumente consumo de vegetais nas refeições")
        if family_history == 'yes':
            recommendations.append("👨‍👩‍👧 Histórico familiar positivo de obesidade aumenta o risco; monitoramento adicional é recomendado")

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

# ============================================================================
# TAB 3: SOBRE O MODELO
# ============================================================================
with tab3:
    st.header("ℹ️ Sobre o Modelo")
    col_model1, col_model2 = st.columns(2)
    with col_model1:
        st.markdown("""
        ### 🤖 Abordagem de Treinamento
        
        O melhor modelo foi selecionado entre Logistic Regression, Random Forest e Gradient Boosting.
        Este modelo foi treinado sem dados antropométricos (altura, peso, IMC) para funcionar como triagem.
        """)
    with col_model2:
        if metadata is not None:
            best_name = metadata.get('best_model_name', 'Modelo não informado')
            best_accuracy = metadata.get('best_accuracy', None)
            model_infos = metadata.get('models', {})
            model_table = "### 📈 Desempenho\n\n| Modelo | Test Accuracy | CV Mean | CV Std |\n|---------|--------------|---------|--------|\n"
            for model_name, info in model_infos.items():
                model_table += f"| {model_name} | {info['accuracy']*100:.2f}% | {info['cv_mean']*100:.2f}% | {info['cv_std']*100:.2f}% |\n"
            st.markdown(model_table)
            if best_accuracy is not None:
                st.write(f"**Melhor modelo selecionado:** {best_name}")
                st.write(f"**Acurácia no teste:** {best_accuracy*100:.2f}%")
        else:
            st.markdown("""
            ### 📈 Desempenho
            
            Modelo final salvo em `models/best_obesity_model.pkl`.
            """)

# ============================================================================
# TAB 4: DICIONÁRIO DE DADOS
# ============================================================================
with tab4:
    st.header("📖 Dicionário de Dados")
    data_dict = {
        "Sexo": "Sexo (Feminino/Masculino)",
        "Age": "Idade em anos (14-61)",
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

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
    <div style='text-align: center; color: #999; padding: 20px;'>
        <p>Sistema de Previsão de Obesidade | Postech - Tech Challenge 4</p>
        <p>Desenvolvido com Python, Streamlit e Machine Learning</p>
    </div>
""", unsafe_allow_html=True)
