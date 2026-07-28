import joblib
import numpy as np
from pathlib import Path

model = joblib.load('models/best_obesity_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')
print('model type', type(model))
print('model repr', repr(model))
print('has predict_proba', hasattr(model, 'predict_proba'))
print('classes', getattr(model, 'classes_', None))
print('scaler mean len', len(getattr(scaler, 'mean_', [])))
print('fn len', len(feature_names))
print('feature_names sample', feature_names[:10])

x = np.zeros((1, len(feature_names)))
xs = scaler.transform(x)
print('x scaled first5', xs[0,:5])
print('min max scaled', xs.min(), xs.max())
p = model.predict_proba(xs)[0]
print('prob', p)
print('sum', p.sum())
print('prob * 100', p*100)
print('prob type', type(p), 'shape', p.shape)
