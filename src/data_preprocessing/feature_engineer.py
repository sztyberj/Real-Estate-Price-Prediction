import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import toml
import warnings
from src.utils.logging_config import logger

pd.set_option('display.float_format', lambda x: f'{x:.4f}')
pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings('ignore')

DATA_DIR = "data/processed"

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, config = None):
        self.config = dict(config) if config is not None else {}

        fe_config = self.config.get('feature_engineering', {})

        self.freq_encode_cols = fe_config.get('frequency_encode_cols', [])
        self.encodings = {}

        self.one_hot_encode_cols = fe_config.get('one_hot_encode_cols', [])
        self.drop_original_onehot = fe_config.get('drop_original_onehot', True)
        self.categories_ = {}

        self.binary_reduce_cols = dict(fe_config.get('binary_reduce_cols', {}))

        self.cols_to_drop_final = fe_config.get('columns_to_drop_final', [])

        self.top_building_types = fe_config.get('top_building_types', [])

    @staticmethod
    def save_to_csv(df: pd.DataFrame , version: str):
        logger.info(f"FeatureEngineer: save to .csv as {version} version")
        df.to_csv(f"{DATA_DIR}/v{version}_{datetime.today().strftime('%Y_%m_%d')}.csv", sep=";", index=False)

    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fit")
        temp_X_for_fit = X.copy()

        #Preprocessing
        if 'building_type' in self.one_hot_encode_cols and 'building_type' in temp_X_for_fit.columns:
            temp_X_for_fit['building_type'] = temp_X_for_fit['building_type'].apply(
                lambda x: x if x in self.top_building_types else 'other'
            )
            logger.info("FeatureEngineer: 'building_type' reduced for fit purposes (to learn correct OHE categories).")

        #Frequency encoding
        for col in self.freq_encode_cols:
            if col in temp_X_for_fit:
                freq_map = temp_X_for_fit[col].value_counts(normalize=True).to_dict()
                self.encodings[col] = freq_map
                temp_X_for_fit[f'{col}_freq'] = temp_X_for_fit[col].map(self.encodings[col]).fillna(0)
                logger.info(f"FeatureEngineer: Learned frequency map for '{col}'.")

        #One hot encoding
        for col in self.one_hot_encode_cols:
            if col in temp_X_for_fit.columns:
                self.categories_[col] = pd.Categorical(temp_X_for_fit[col]).categories.tolist()
                logger.info(f"FeatureEngineer: Learned categories for '{col}': {self.categories_[col]}.")

        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transform")
        df_transformed = X.copy()
        #1. elevator_in_desc
        if 'description' in df_transformed.columns:
            has_elevator = df_transformed['description'].str.contains(r'\bwinda\w*\b', case=False, na=False)
            df_transformed.loc[has_elevator, 'elevator'] = 1
            df_transformed['elevator'] = df_transformed['elevator'].replace({'Tak': 1, 'Nie': 0})
            df_transformed['elevator'] = df_transformed['elevator'].fillna(0).astype(int)
            logger.info("FeatureEngineer: 'elevator' feature created.")

        #2. has_balcony_in_desc
        if 'description' in df_transformed.columns:
            has_balcony = df_transformed['description'].str.contains(r'\b(balkon\w*|taras\w*)\b', case=False, na=False)
            df_transformed.loc[has_balcony, 'balcony'] = 1
            df_transformed['balcony'] = df_transformed['balcony'].fillna(0).astype(int)
            logger.info("FeatureEngineer: 'balcony' feature created.")

        #3. has_garage_in_desc
        if 'description' in df_transformed.columns:
            has_garage = df_transformed['description'].str.contains(r'\bgaraż\w*\b', case=False, na=False)
            df_transformed.loc[has_garage, 'garage'] = 1
            df_transformed['garage'] = df_transformed['garage'].fillna(0).astype(int)
            logger.info("FeatureEngineer: 'garage' feature created.")

        #4. has_furniture_in_desc
        if 'description' in df_transformed.columns:
            has_furniture = df_transformed['description'].str.contains(r'\bmeble\w*\b', case=False, na=False)
            df_transformed.loc[has_furniture, 'furnished'] = 1
            df_transformed['furnished'] = df_transformed['furnished'].replace({'Tak': 1, 'Nie': 0})
            df_transformed['furnished'] = df_transformed['furnished'].fillna(0).astype(int)
            logger.info("FeatureEngineer: 'furnished' feature created.")

        #5. create room_per_area col
        if "rooms" in df_transformed.columns and "area" in df_transformed.columns and not df_transformed['area'].eq(0).any():
            df_transformed["rooms_per_area"] = (df_transformed["rooms"] / df_transformed["area"]).astype(float)
            # Handle potential division by zero results
            df_transformed['rooms_per_area'].replace([np.inf, -np.inf], np.nan, inplace=True)
            df_transformed['rooms_per_area'].fillna(0, inplace=True) # Fill with 0 or median/mean
            logger.info("FeatureEngineer: 'rooms_per_area' feature created.")

        #6. reduce_to_binary
        for col, positive_val in self.binary_reduce_cols.items():
            if col in df_transformed.columns:
                df_transformed[col] = (df_transformed[col] == positive_val).astype(int)
                logger.info(f"FeatureEngineer: Reduced '{col}' to binary.")

        #7. reduce_building_type
        if 'building_type' in df_transformed.columns:
            top_types = self.top_building_types
            df_transformed['building_type'] = df_transformed['building_type'].apply(
                lambda x: x if x in top_types else 'other'
            )
            logger.info("FeatureEngineer: 'building_type' reduced.")

        #9. frequency_encoding
        for col in self.freq_encode_cols:
            if col in df_transformed.columns and col in self.encodings:
                df_transformed[f'{col}_freq'] = df_transformed[col].map(self.encodings[col]).fillna(0)
                logger.info(f"FeatureEngineer: Applied frequency encoding for '{col}'.")

        #12. drop columns
        cols_to_actually_drop = [c for c in self.cols_to_drop_final if c in df_transformed.columns]
        if cols_to_actually_drop:
            df_transformed = df_transformed.drop(cols_to_actually_drop, axis=1)
            logger.info(f"FeatureEngineer: Dropped columns: {cols_to_actually_drop}")

        logger.info("FeatureEngineer: Transformation complete.")

        return df_transformed

if __name__ == "__main__":
    from src.eda.data_reader import DataReader
    from src.data_preprocessing.data_processor import DataProcessor

    
    with open("config.toml", 'r') as f:
        config = toml.load(f)
    
    reader = DataReader()
    df = reader.read()
    
    data_processor = DataProcessor(config)
    data_processor.fit(df)
    df = data_processor.transform(df)
    data_processor.save_to_csv(df, version="T")

    engineer = FeatureEngineer(config)

    X = df.drop('price', axis=1)
    y = df['price']

    engineer.fit(X)
    X = engineer.transform(X)
    print(X)