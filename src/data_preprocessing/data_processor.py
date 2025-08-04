import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
import toml
import warnings
from src.utils.logging_config import logger

pd.set_option('display.float_format', lambda x: f'{x:.4f}')
pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings('ignore')

DATA_DIR = "data/cleaned"

with open("config.toml", 'r') as f:
    config = toml.load(f)

class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_check=None, columns_to_clean=None, 
                 columns_to_drop=None, floor_map=None, outlier_lower=0.05, 
                 outlier_upper=0.95, **kwargs):

        self.columns_to_check = columns_to_check if columns_to_check is not None else []
        self.columns_to_clean = columns_to_clean if columns_to_clean is not None else []
        self.columns_to_drop = columns_to_drop if columns_to_drop is not None else []
        self.floor_map = floor_map if floor_map is not None else {}
        self.outlier_lower = outlier_lower
        self.outlier_upper = outlier_upper
        
        self.year_built_median_ = None
        self.rent_median_ = None
        self.outlier_thresholds_ = {}
    
    @staticmethod
    def save_to_csv(df: pd.DataFrame, version: str):
        logger.info(f"DataProcessor: save to .csv as {version} version")
        Path(DATA_DIR).mkdir(exist_ok=True)
        df.to_csv(f"{DATA_DIR}/v{version}_{datetime.today().strftime('%Y_%m_%d')}.csv", sep=";", index=False)

    def fit(self, X, y=None):
        logger.info("DataProcessor: Fitting...")
        temp_df = X.copy()

        # Learn median for year_built
        if 'year_built' in temp_df.columns:
            self.year_built_median_ = temp_df['year_built'].median()
            logger.info(f"DataProcessor: Learned median for 'year_built': {self.year_built_median_}")

        # Learn median for rent
        if 'rent' in temp_df.columns:
            self.rent_median_ = temp_df['rent'].median()
            logger.info(f"DataProcessor: Learned median for 'rent': {self.rent_median_}")

        # Learn outlier thresholds for numerical columns
        for col in self.columns_to_clean:
            if col in temp_df.columns:
                # Convert to numeric first
                temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                if temp_df[col].notna().sum() > 0:  # Only if we have valid data
                    lower_threshold = temp_df[col].quantile(self.outlier_lower)
                    upper_threshold = temp_df[col].quantile(self.outlier_upper)
                    self.outlier_thresholds_[col] = (lower_threshold, upper_threshold)
                    logger.info(f"DataProcessor: Learned outlier thresholds for '{col}': {lower_threshold:.2f} - {upper_threshold:.2f}")

        return self

    def transform(self, X):
        logger.info("DataProcessor: Transforming...")
        df_transformed = X.copy()

        # Drop specified columns
        cols_to_drop = [col for col in self.columns_to_drop if col in df_transformed.columns]
        if cols_to_drop:
            df_transformed.drop(columns=cols_to_drop, inplace=True)
            logger.info(f"DataProcessor: Dropped columns: {cols_to_drop}")

        # Clean numerical columns
        for col in self.columns_to_clean:
            if col in df_transformed.columns:
                df_transformed[col] = pd.to_numeric(df_transformed[col], errors='coerce')
                
                if col in self.outlier_thresholds_:
                    lower_thresh, upper_thresh = self.outlier_thresholds_[col]
                    
                    # Count outliers before clipping
                    outliers_lower = (df_transformed[col] < lower_thresh).sum()
                    outliers_upper = (df_transformed[col] > upper_thresh).sum()
                    
                    # Clip values instead of removing rows
                    df_transformed[col] = df_transformed[col].clip(lower=lower_thresh, upper=upper_thresh)
                    
                    if outliers_lower + outliers_upper > 0:
                        logger.info(f"DataProcessor: Clipped {outliers_lower + outliers_upper} outliers in '{col}' "
                                  f"(lower: {outliers_lower}, upper: {outliers_upper})")

        # Handle missing values
        if 'year_built' in df_transformed.columns and self.year_built_median_ is not None:
            missing_count = df_transformed['year_built'].isnull().sum()
            if missing_count > 0:
                df_transformed['year_built'].fillna(self.year_built_median_, inplace=True)
                logger.info(f"DataProcessor: Filled {missing_count} missing values in 'year_built' with median: {self.year_built_median_}")

        if 'rent' in df_transformed.columns and self.rent_median_ is not None:
            missing_count = df_transformed['rent'].isnull().sum()
            if missing_count > 0:
                df_transformed['rent'].fillna(self.rent_median_, inplace=True)
                logger.info(f"DataProcessor: Filled {missing_count} missing values in 'rent' with median: {self.rent_median_}")

        # Apply floor mapping
        if 'floor' in df_transformed.columns and self.floor_map:
            for floor_name, floor_value in self.floor_map.items():
                mask = df_transformed['floor'].astype(str).str.lower() == floor_name.lower()
                df_transformed.loc[mask, 'floor'] = floor_value
            
            # Convert floor to numeric
            df_transformed['floor'] = pd.to_numeric(df_transformed['floor'], errors='coerce')
            logger.info(f"DataProcessor: Applied floor mapping: {self.floor_map}")

        # Handle categorical columns - fill missing values with 'unknown'
        for col in self.columns_to_check:
            if col in df_transformed.columns:
                missing_count = df_transformed[col].isnull().sum()
                if missing_count > 0:
                    df_transformed[col].fillna('unknown', inplace=True)
                    logger.info(f"DataProcessor: Filled {missing_count} missing values in '{col}' with 'unknown'")

        # Reset index (though no rows should be removed now)
        df_transformed.reset_index(drop=True, inplace=True)
        
        logger.info("DataProcessor: Transformation complete.")
        return df_transformed