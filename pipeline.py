import pandas as pandas
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import toml
import sys
from pathlib import Path
from src.utils.logging_config import logger
import json
from src.eda.data_reader import DataReader
from src.data_preprocessing.data_processor import DataProcessor 
from src.data_preprocessing.feature_engineer import FeatureEngineer

THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[0]
print(ROOT_DIR)
sys.path.append(str(ROOT_DIR))

with open(ROOT_DIR / "config.toml", 'r') as f:
    config = toml.load(f)
    config = json.loads(json.dumps(config))

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

df = reader.read()

data_processor.fit(df)

df = data_processor.transform(df)

data_shuffled = df.sample(frac=1, random_state=14).reset_index(drop=True)

n = len(data_shuffled)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

df_train = data_shuffled.iloc[:train_end]
df_val   = data_shuffled.iloc[train_end:val_end]
df_test  = data_shuffled.iloc[val_end:]

#train set
y_train = df_train.iloc[:, 0].to_numpy()
X_train_df = df_train.iloc[:, 1:]

#val/dev set
y_val = df_val.iloc[:, 0].to_numpy()
X_val_df = df_val.iloc[:, 1:]

#test/test set
y_test = df_test.iloc[:, 0].to_numpy()
X_test_df = df_test.iloc[:, 1:]

logger.info(f"Train size: {X_train_df.shape[0], y_train.shape[0]}")
logger.info(f"Val size: {X_val_df.shape[0], y_val.shape[0]}")
logger.info(f"Test size: {X_test_df.shape[0], y_test.shape[0]}")

feature_engineer.fit(X_train_df)

X_train_df = feature_engineer.transform(X_train_df)
X_val_df = feature_engineer.transform(X_val_df)
X_test_df = feature_engineer.transform(X_test_df)

logger.info("Scale Y")
y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

logger.info("Train XGBRegressor")
xgb_model.fit(X_train_df, y_train_scaled)

y_val_pred = xgb_model.predict(X_val_df)
val_mse = mean_squared_error(y_val_scaled, y_val_pred)
val_r2 = r2_score(y_val_scaled, y_val_pred)

logger.info(f"MSE Val: {val_mse:.4f}")
logger.info(f"R2 Val: {val_r2:.4f}")

joblib.dump(full_pipeline, 'real_estate_model_pipeline.joblib')
logger.info("\nFull pipeline saved to 'real_estate_model_pipeline.joblib'")