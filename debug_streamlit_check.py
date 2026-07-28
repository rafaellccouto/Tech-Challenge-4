import urllib.request
import urllib.error
import joblib
import pandas as pd
import numpy as np

# Load artifacts
model = joblib.load('models/best_obesity_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Build a sample input with feature names matching the model
input_data = {
    'Gender': 1,
    'Age': 30,
    'family_history': 1,
    'FAVC': 1,
    'FCVC': 2,
    'NCP': 3,
    'SMOKE': 0,
    'CH2O': 2,
    'SCC': 1,
    'FAF': 1,
    'TUE': 1,
    'MTRANS_Automobile': 1,
    'MTRANS_Bike': 0,
    'MTRANS_Motorbike': 0,
    'MTRANS_Public_Transportation': 0,
    'MTRANS_Walking': 0,
    'CAEC_Always': 0,
    'CAEC_Frequently': 0,
    'CAEC_Sometimes': 1,
    'CAEC_no': 0,
    'CALC_Always': 0,
    'CALC_Frequently': 0,
    'CALC_Sometimes': 1,
    'CALC_no': 0,
}

df = pd.DataFrame([input_data])
df = df[[c for c in feature_names]]
X = scaler.transform(df.values)
pred = model.predict(X)[0]
prob = model.predict_proba(X)[0].astype(float)
prob = np.clip(prob, 0.0, None)
prob = prob / np.sum(prob)
print('LOCAL_SUM:', prob.sum())
print('LOCAL_PROB:', prob)
print('LOCAL_LABEL:', label_encoder.inverse_transform([pred])[0])

# Remote URL check
url = 'https://tech-challenge-4-previsao-obesidade.streamlit.app/'
try:
    with urllib.request.urlopen(url, timeout=30) as response:
        body = response.read(2048).decode('utf-8', errors='replace')
        print('REMOTE_STATUS:', response.status)
        start = body.find('<title>')
        end = body.find('</title>', start)
        if start != -1 and end != -1:
            print('REMOTE_TITLE:', body[start+7:end].strip())
        else:
            print('REMOTE_BODY_START:', body[:200].replace('\n', ' '))
except urllib.error.HTTPError as e:
    print('REMOTE_HTTP_ERROR:', e.code, e.reason)
    print(e.read(2048).decode('utf-8', errors='replace'))
except Exception as e:
    print('REMOTE_ERROR:', repr(e))
