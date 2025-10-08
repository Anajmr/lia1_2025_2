# cancelar.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Previsão de Cancelamento", layout="wide")

# -------------------------
# Carrega dados e treina modelo
# -------------------------
@st.cache_data
def load_data_and_model(path="cancelamentos.csv"):
    data = pd.read_csv(path)

    # Identifica colunas categóricas e numéricas
    cat_cols = data.select_dtypes(include=['object']).columns.drop(['customerID','Churn'])
    num_cols = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']

    # Preenche NaN nas categóricas com valor mais frequente
    for c in cat_cols:
        data[c] = data[c].fillna(data[c].mode()[0])

    # Converte TotalCharges para numérico
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

    # Encoder com tratamento de valores desconhecidos
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    data_encoded = data.copy()
    data_encoded[cat_cols] = encoder.fit_transform(data[cat_cols])

    # Define X e y
    y = data_encoded['Churn'].map({'No':0,'Yes':1})
    X = data_encoded.drop(['customerID','Churn'], axis=1)

    # Treina modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = CategoricalNB()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return encoder, model, float(accuracy), data, list(cat_cols), list(num_cols), list(X.columns)

encoder, model, accuracy, data_hist, cat_cols, num_cols, feature_order = load_data_and_model()
st.write(f"Acurácia do modelo com dados históricos: **{accuracy:.2f}**")

st.subheader("📝 Insira os dados do cliente para previsão")

# -------------------------
# Inputs numéricos
# -------------------------
senior_sel = st.selectbox("Cliente idoso:", ("Não", "Sim"))
senior_numeric = 1 if senior_sel == "Sim" else 0

tenure = st.slider("Tempo de assinatura (meses):", 0, 600, value=12, step=1)
monthly_charges = st.number_input("Valor mensal (R$):", min_value=0.0, step=0.1, value=50.0, format="%.2f")

input_model = {}
input_model['SeniorCitizen'] = senior_numeric
input_model['tenure'] = tenure
input_model['MonthlyCharges'] = monthly_charges
input_model['TotalCharges'] = tenure * monthly_charges  # calculado internamente

# -------------------------
# Categorias em português
# -------------------------
categorical_options_pt = {
    'gender': ['Feminino', 'Masculino'],
    'Partner': ['Sim', 'Não'],
    'Dependents': ['Sim', 'Não'],
    'PhoneService': ['Sim', 'Não'],
    'MultipleLines': ['Sem serviço telefônico', 'Não', 'Sim'],
    'InternetService': ['DSL (banda larga)', 'Fibra óptica', 'Sem internet contratada'],
    'OnlineSecurity': ['Sim', 'Não', 'Sem internet'],
    'OnlineBackup': ['Sim', 'Não', 'Sem internet'],
    'DeviceProtection': ['Sim', 'Não', 'Sem internet'],
    'TechSupport': ['Sim', 'Não', 'Sem internet'],
    'StreamingTV': ['Sim', 'Não', 'Sem internet'],
    'StreamingMovies': ['Sim', 'Não', 'Sem internet'],
    'Contract': ['Mensal', 'Anual', 'Bienal'],
    'PaperlessBilling': ['Sim', 'Não'],
    'PaymentMethod': ['Cheque eletrônico', 'Cheque enviado', 'Transferência bancária', 'Cartão de crédito']
}

categorical_options_en = {
    'gender': {'Feminino':'Female','Masculino':'Male'},
    'Partner': {'Sim':'Yes','Não':'No'},
    'Dependents': {'Sim':'Yes','Não':'No'},
    'PhoneService': {'Sim':'Yes','Não':'No'},
    'MultipleLines': {'Sem serviço telefônico':'No phone service','Não':'No','Sim':'Yes'},
    'InternetService': {'DSL (banda larga)':'DSL','Fibra óptica':'Fiber optic','Sem internet contratada':'No'},
    'OnlineSecurity': {'Sim':'Yes','Não':'No','Sem internet':'No internet service'},
    'OnlineBackup': {'Sim':'Yes','Não':'No','Sem internet':'No internet service'},
    'DeviceProtection': {'Sim':'Yes','Não':'No','Sem internet':'No internet service'},
    'TechSupport': {'Sim':'Yes','Não':'No','Sem internet':'No internet service'},
    'StreamingTV': {'Sim':'Yes','Não':'No','Sem internet':'No internet service'},
    'StreamingMovies': {'Sim':'Yes','Não':'No','Sem internet':'No internet service'},
    'Contract': {'Mensal':'Month-to-month','Anual':'One year','Bienal':'Two year'},
    'PaperlessBilling': {'Sim':'Yes','Não':'No'},
    'PaymentMethod': {'Cheque eletrônico':'Electronic check','Cheque enviado':'Mailed check','Transferência bancária':'Bank transfer (automatic)','Cartão de crédito':'Credit card (automatic)'}
}

# -------------------------
# Dropdowns com valor padrão do histórico
# -------------------------
for c in categorical_options_pt.keys():
    valor_padrao = data_hist[c].mode()[0]
    pt_padrao = [k for k,v in categorical_options_en[c].items() if v==valor_padrao][0] if valor_padrao in categorical_options_en[c].values() else categorical_options_pt[c][0]
    index_padrao = categorical_options_pt[c].index(pt_padrao)
    escolha_pt = st.selectbox(f"{c.replace('_',' ')}", categorical_options_pt[c], index=index_padrao)
    input_model[c] = categorical_options_en[c][escolha_pt]

# -------------------------
# Previsão
# -------------------------
if st.button("Prever Cancelamento"):
    df_model = pd.DataFrame([input_model], columns=feature_order)

    # Garantir que todas as categóricas estão como string e substituir desconhecidos
    for c in cat_cols:
        df_model[c] = df_model[c].astype(str)
        if df_model[c][0] not in encoder.categories_[cat_cols.index(c)]:
            df_model[c][0] = data_hist[c].mode()[0]

    # Transformar categorias e remover valores negativos
    df_model[cat_cols] = encoder.transform(df_model[cat_cols])
    df_model[cat_cols] = df_model[cat_cols].clip(lower=0)

    # Garantir que numéricos estão float
    for n in num_cols:
        df_model[n] = pd.to_numeric(df_model[n], errors='coerce').astype(float)

    pred = model.predict(df_model)[0]
    proba = model.predict_proba(df_model)[0][1]

    # Classificação textual
    if proba < 0.3:
        resultado = "BAIXA CHANCE DE CANCELAMENTO"
        cor = "#28a745"
    elif proba < 0.7:
        resultado = "CHANCE MÉDIA DE CANCELAMENTO"
        cor = "#ffc107"
    else:
        resultado = "ALTA CHANCE DE CANCELAMENTO"
        cor = "#dc3545"

    st.subheader("Resultado da Previsão")
    st.markdown(
        f"<h1 style='color:{cor}; font-size:48px; text-align:center'>{resultado}</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='text-align:center; font-size:20px;'>Probabilidade de cancelamento: <b>{proba*100:.1f}%</b></p>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='text-align:center; font-size:16px;'>Total Charges calculado: R$ {input_model['TotalCharges']:.2f}</p>",
        unsafe_allow_html=True
    )
    st.progress(int(proba*100))
