import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import toml
from pathlib import Path
from src.utils.logging_config import logger
import json
from src.eda.data_reader import DataReader
from src.data_preprocessing.data_processor import DataProcessor 
from src.data_preprocessing.feature_engineer import FeatureEngineer


def target_cleaner(y) -> pd.DataFrame:
    y['price'] = pd.to_numeric(y['price'], errors='coerce')
    y.dropna(subset=['price'], inplace=True)

    return y

with open(r"C:\Users\Jakub\Real Estate Price Prediction\config.toml", 'r') as f:
    config = toml.load(f)
    config = json.loads(json.dumps(config))

dp_params = config.get('data_processing', {})
fe_params = config.get('feature_engineering', {})
model_params = config.get('model_params', {})
pipeline_params = config.get('pipeline', {})
version = pipeline_params['version']
columns_to_scale = fe_params.get("columns_to_scale", [])

reader = DataReader()
data = reader.read()

data = target_cleaner(data)

X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), columns_to_scale)
    ],
    remainder='passthrough'
)

X_pipeline = Pipeline(steps=[
    ('preprocessor', DataProcessor(config)),
    ('feature_engineer', FeatureEngineer(config)),
    ('scaler', scaler)
])

X_train_processed = X_pipeline.fit_transform(X_train)
X_test_processed = X_pipeline.transform(X_test)
processed_columns = X_pipeline.named_steps['scaler'].get_feature_names_out()

X_train = pd.DataFrame(X_train_processed, index=X_train.index[:len(X_train_processed)], columns=processed_columns)
X_test = pd.DataFrame(X_test_processed, index=X_test.index[:len(X_test_processed)], columns=processed_columns)

y_train_synced = y_train.loc[X_train.index]
y_test_synced = y_test.loc[X_test.index]

y_train = y_train_synced.ravel()
y_test = y_test_synced.ravel()

model = xgb.XGBRegressor(**model_params)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred[:15].ravel())

X_pipeline.named_steps['preprocessor'].set_production_mode()

# Save artefacts
logger.info("Saving components...")
Path("models").mkdir(exist_ok=True)
joblib.dump(X_pipeline, f'models/X_pipeline_{version}.joblib')
joblib.dump(model, f'models/model_{version}.joblib')

logger.info(f"X_pipeline and model saved successfully.")