import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor
import joblib
import toml
from pathlib import Path
from src.utils.logging_config import logger
import json
import matplotlib.pyplot as plt
import seaborn as sns
from src.eda.data_reader import DataReader
from src.data_preprocessing.data_processor import DataProcessor 
from src.data_preprocessing.feature_engineer import FeatureEngineer

# Load config
with open("config.toml", 'r') as f:
    config = toml.load(f)
    config = json.loads(json.dumps(config))

pipe_config = config.get('model_params', {})
version = pipe_config.get('version', "")
TARGET_COLUMN = 'price'
dp_params = config.get('data_processing', {})

logger.info(f"[START] Pipeline v{version} with Cross-Validation")

reader = DataReader()
df_raw = reader.read()

# Safe cleaning and prep data
logger.info(f"Performing safe, stateless cleaning of the target column '{TARGET_COLUMN}'")
df_raw['price'] = pd.to_numeric(df_raw['price'], errors='coerce')
initial_rows = len(df_raw)
df_raw.dropna(subset=[TARGET_COLUMN], inplace=True)
dropped_rows = initial_rows - len(df_raw)
if dropped_rows > 0:
    logger.info(f"Dropped {dropped_rows} rows due to missing target.")

logger.info("Performing outlier removal before train-test split to maintain data alignment")

# Apply outlier removal based on config parameters
outlier_lower = dp_params.get('outlier_lower', 0.05)
outlier_upper = dp_params.get('outlier_upper', 0.95)
columns_to_clean = dp_params.get('columns_to_clean', [])

for col in columns_to_clean:
    if col in df_raw.columns and col != TARGET_COLUMN:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        if df_raw[col].notna().sum() > 0:
            lower_threshold = df_raw[col].quantile(outlier_lower)
            upper_threshold = df_raw[col].quantile(outlier_upper)
            
            before_count = len(df_raw)
            df_raw = df_raw[
                (df_raw[col].isnull()) |  # Keep null values for now
                ((df_raw[col] >= lower_threshold) & (df_raw[col] <= upper_threshold))
            ]
            removed_count = before_count - len(df_raw)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} outliers from '{col}' before train-test split")

# Reset index after row removal
df_raw.reset_index(drop=True, inplace=True)
logger.info(f"Final dataset size after outlier removal: {len(df_raw)} rows")

# Split data
data_shuffled = df_raw.sample(frac=1, random_state=14).reset_index(drop=True)

y = data_shuffled[TARGET_COLUMN]
X = data_shuffled.drop(columns=[TARGET_COLUMN])

n = len(data_shuffled)
train_val_end = int(0.85 * n) # 85% on training and CV

X_train_val, X_test = X.iloc[:train_val_end], X.iloc[train_val_end:]
y_train_val, y_test = y.iloc[:train_val_end], y.iloc[train_val_end:]

logger.info(f"Train/Val set size: {len(X_train_val)}, Test set size: {len(X_test)}")

logger.info("Scaling target variable 'y'")
y_scaler = StandardScaler()
y_train_val_scaled = y_scaler.fit_transform(y_train_val.to_numpy().reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.to_numpy().reshape(-1, 1)).ravel()

fe_params = config.get('feature_engineering', {})

dp_params_for_pipeline = dp_params.copy()
dp_params_for_pipeline['outlier_lower'] = 0.0  # Disable outlier removal
dp_params_for_pipeline['outlier_upper'] = 1.0  # Disable outlier removal

data_processor = DataProcessor(**dp_params_for_pipeline) 
feature_engineer = FeatureEngineer(**fe_params)
xgb_model = XGBRegressor(n_estimators=100, random_state=14)

full_pipeline = Pipeline(steps=[
    ('data_processing', data_processor),
    ('feature_engineering', feature_engineer),
    ('model', xgb_model)
])

logger.info(f"Pełny pipeline do ewaluacji: {full_pipeline}")

# Cross-Validation)
logger.info("Step 1: Performing Cross-Validation")
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=14)

cv_results = cross_validate(
    full_pipeline, 
    X_train_val, 
    y_train_val_scaled, 
    cv=cv_strategy,
    scoring=('r2', 'neg_mean_squared_error'),
    return_train_score=True
)

# Show CV results
mean_cv_r2 = np.mean(cv_results['test_r2'])
std_cv_r2 = np.std(cv_results['test_r2'])
mean_cv_mse = -np.mean(cv_results['test_neg_mean_squared_error'])

logger.info(f"Cross-Validation R2: {mean_cv_r2:.4f} +/- {std_cv_r2:.4f}")
logger.info(f"Cross-Validation MSE: {mean_cv_mse:.4f}")

# Training final model
logger.info("Step 2: Fitting final model on the entire training set")
full_pipeline.fit(X_train_val, y_train_val_scaled)

logger.info("Step 3: Feature Importance Analysis")

final_model = full_pipeline.named_steps['model']
preprocessor_pipeline = Pipeline(full_pipeline.steps[:-1])

X_train_transformed = preprocessor_pipeline.transform(X_train_val)
feature_names = X_train_transformed.columns.tolist()

importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

logger.info("Top 20 najważniejszych cech:")
logger.info(f"\n{feature_importance_df.head(20)}")

# Feature importance plot
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig(f'reports/feature_importance_{version}.png')
logger.info(f"Wykres istotności cech zapisany w 'reports/feature_importance_{version}.png'")

# Final evaluation on test set
logger.info("Step 4: Evaluating the final model on the hold-out test set")
y_test_pred = full_pipeline.predict(X_test)

test_mse = mean_squared_error(y_test_scaled, y_test_pred)
test_r2 = r2_score(y_test_scaled, y_test_pred)

logger.info(f"Final Test MSE: {test_mse:.4f}")
logger.info(f"Final Test R2: {test_r2:.4f}")

# Save artefacts
logger.info("Saving components...")
Path("models").mkdir(exist_ok=True)
joblib.dump(full_pipeline, f'models/real_estate_full_pipeline_{version}.joblib')
joblib.dump(y_scaler, f'models/y_scaler_{version}.joblib')
logger.info(f"Trained pipeline and y_scaler saved successfully.")