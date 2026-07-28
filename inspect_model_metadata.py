import joblib
from pathlib import Path
metadata_path = Path('models/model_metadata.pkl')
metadata = joblib.load(metadata_path)
print(metadata)
