import pandas as pandas
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import toml
from pathlib import Path
from src.utils.logging_config import logger
import json
from src.eda.data_reader import DataReader
from src.data_preprocessing.data_processor import DataProcessor 
from src.data_preprocessing.feature_engineer import FeatureEngineer

with open("config.toml", 'r') as f:
    config = toml.load(f)
    config = json.loads(json.dumps(config))

pipe_config = config.get('model_params', {})
version = pipe_config.get('version', "")

logger.info("[START] Pipeline")

reader = DataReader()
data_processor = DataProcessor(config)
feature_engineer = FeatureEngineer(config)
xgb_model = XGBRegressor(n_estimators=100, random_state=14)

full_pipeline = Pipeline(steps=[
    ('data_processing', data_processor),
    ('feature_engineering', feature_engineer),
    ('model', xgb_model)
])

logger.info(str(full_pipeline))

df_raw = reader.read()

logger.info("Step 1: Initial data processing with DataProcessor")
data_processor.fit(df_raw)
df_processed = data_processor.transform(df_raw)
logger.info("Initial data processing complete. 'price' column is now clean.")

data_shuffled = df_processed.sample(frac=1, random_state=14).reset_index(drop=True)

n = len(data_shuffled)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

df_train = data_shuffled.iloc[:train_end]
df_val = data_shuffled.iloc[train_end:val_end]
df_test = data_shuffled.iloc[val_end:]

TARGET_COLUMN = 'price'
y_train = df_train[TARGET_COLUMN]
X_train = df_train.drop(columns=[TARGET_COLUMN])

y_val = df_val[TARGET_COLUMN]
X_val = df_val.drop(columns=[TARGET_COLUMN])

logger.info("Scaling target variable 'y'")
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))
y_val_scaled = y_scaler.transform(y_val.to_numpy().reshape(-1, 1))

feature_model_pipeline = Pipeline(steps=[
    ('feature_engineering', feature_engineer),
    ('model', xgb_model)
])

logger.info("Step 4: Fitting the feature engineering and model pipeline")
feature_model_pipeline.fit(X_train, y_train_scaled.ravel())

logger.info("Evaluating the model")
y_val_pred = feature_model_pipeline.predict(X_val)
val_mse = mean_squared_error(y_val_scaled, y_val_pred)
val_r2 = r2_score(y_val_scaled, y_val_pred)

logger.info(f"MSE Val: {val_mse:.4f}")
logger.info(f"R2 Val: {val_r2:.4f}")

logger.info("Saving components...")

trained_components = {
    'data_processor': data_processor,
    'feature_model_pipeline': feature_model_pipeline,
    'y_scaler': y_scaler
}
joblib.dump(trained_components, f'models/real_estate_model_components_{version}.joblib')
logger.info(f"Trained components saved to 'models/real_estate_model_components_{version}.joblib'")
