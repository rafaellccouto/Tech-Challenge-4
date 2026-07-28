import joblib
import numpy as np

model = joblib.load('models/best_obesity_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
feature_names = joblib.load('models/feature_names.pkl')

print('model type:', type(model))
print('has predict_proba:', hasattr(model, 'predict_proba'))
print('classes_:', getattr(model, 'classes_', None))
print('n_classes_:', getattr(model, 'n_classes_', None))
print('n_features_in_:', getattr(model, 'n_features_in_', None))
print('n_outputs_:', getattr(model, 'n_outputs_', None))
print('n_estimators:', getattr(model, 'n_estimators', None))
print('predict_proba type:', type(model.predict_proba))
try:
    print('predict_proba qualname:', model.predict_proba.__qualname__)
    print('predict_proba module:', model.predict_proba.__module__)
except Exception:
    pass
print('label_encoder classes_:', getattr(label_encoder, 'classes_', None))
print('num feature_names:', len(feature_names))
print('first feature_names sample:', feature_names[:10])
print('scaler mean_ len:', len(getattr(scaler, 'mean_', [])))
print('scaler var_ len:', len(getattr(scaler, 'var_', [])))
print('model __dict__ keys:', list(model.__dict__.keys()))

x = np.zeros((1, len(feature_names)))
print('x shape', x.shape)
xs = scaler.transform(x)
print('x scaled sample first5', xs[0, :5])
print('x_scaled min/max', xs.min(), xs.max())

p = model.predict_proba(xs)[0]
print('predict_proba output:', p)
print('sum predict_proba:', p.sum())
print('probabilities * 100:', p * 100)
print('n_estimators:', len(model.estimators_))
for i, est in enumerate(model.estimators_[:5]):
    try:
        pt = est.predict_proba(xs)[0]
        print(f'tree {i} sum:', pt.sum(), 'pt:', pt)
        print(f'tree {i} classes_:', getattr(est, 'classes_', None))
    except Exception as e:
        print(f'tree {i} predict_proba error:', e)

print('model classes_:', model.classes_)
print('estimator classes_ sample:', getattr(model.estimators_[0], 'classes_', None))

# Debug actual app-like input with plausible values
sample = np.zeros((1, len(feature_names)))
# Use 1 sample values by index if needed
# We can't know columns exactly, but if feature_names are in order we can set key names here if we inspect them
print('done')
