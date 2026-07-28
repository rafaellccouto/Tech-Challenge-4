import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATA_PATH = Path('Dados_base/Obesity.csv')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna().copy()

    X = df.drop(columns=['Obesity', 'Height', 'Weight'])
    y = df['Obesity']

    binary_map = {'yes': 1, 'no': 0}
    for col in ['family_history', 'FAVC', 'SMOKE', 'SCC']:
        X[col] = X[col].map(binary_map)

    X['Gender'] = X['Gender'].map({'Female': 0, 'Male': 1})

    X = pd.get_dummies(
        X,
        columns=['CAEC', 'CALC', 'MTRANS'],
        prefix=['CAEC', 'CALC', 'MTRANS'],
        drop_first=False,
    )

    return X, y


def train_models(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        stratify=y_encoded,
        test_size=0.20,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        'Logistic Regression': LogisticRegression(
            penalty='l2', solver='lbfgs', max_iter=2000, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        ),
    }

    results = {}
    for name, model in candidates.items():
        print(f'Training {name}...')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
        )
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
        }
        print(f'  Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
        print(f'  CV mean accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, le, scaler, results


if __name__ == '__main__':
    X, y = load_data()
    (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        label_encoder,
        scaler,
        results,
    ) = train_models(X, y)

    best_name = max(results.keys(), key=lambda name: results[name]['accuracy'])
    best_model = results[best_name]['model']
    best_accuracy = results[best_name]['accuracy']

    print('\nSelected best model:', best_name)
    print(f'Best test accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)')

    joblib.dump(best_model, MODELS_DIR / 'best_obesity_model.pkl')
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    joblib.dump(label_encoder, MODELS_DIR / 'label_encoder.pkl')
    joblib.dump(X.columns.tolist(), MODELS_DIR / 'feature_names.pkl')
    joblib.dump(
        {
            'best_model_name': best_name,
            'best_accuracy': best_accuracy,
            'models': {
                name: {
                    'accuracy': info['accuracy'],
                    'cv_mean': info['cv_mean'],
                    'cv_std': info['cv_std'],
                }
                for name, info in results.items()
            },
        },
        MODELS_DIR / 'model_metadata.pkl',
    )

    for name, info in results.items():
        filename = name.lower().replace(' ', '_') + '_no_anthropometry.pkl'
        joblib.dump(info['model'], MODELS_DIR / filename)

    print('\nArtifacts saved in models/')
    print('  - best_obesity_model.pkl')
    print('  - scaler.pkl')
    print('  - label_encoder.pkl')
    print('  - feature_names.pkl')
    print('  - model_metadata.pkl')
    for name in results:
        print(f'  - {name.lower().replace(" ", "_")}_no_anthropometry.pkl')
