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

with open("config.toml", 'r') as f:
    config = toml.load(f)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, frequency_encode_cols=None, one_hot_encode_cols=None, drop_original_onehot=True,
                 binary_reduce_cols=None, columns_to_drop_final=None, top_building_types=None, columns_to_scale=None):
        
        self.frequency_encode_cols = frequency_encode_cols if frequency_encode_cols is not None else []
        self.one_hot_encode_cols = one_hot_encode_cols if one_hot_encode_cols is not None else []
        self.binary_reduce_cols = binary_reduce_cols if binary_reduce_cols is not None else {}
        self.columns_to_drop_final = columns_to_drop_final if columns_to_drop_final is not None else []
        self.top_building_types = top_building_types if top_building_types is not None else []
        self.columns_to_scale = columns_to_scale if columns_to_scale is not None else []

        self.drop_original_onehot = drop_original_onehot
        
        self.encodings_ = {}
        self.categories_ = {}
        self.scaler_ = StandardScaler()
        self.columns_for_scaling_ = []

    @staticmethod
    def save_to_csv(df: pd.DataFrame, version: str):
        logger.info(f"FeatureEngineer: save to .csv as {version} version")
        Path(DATA_DIR).mkdir(exist_ok=True)
        df.to_csv(f"{DATA_DIR}/v{version}_{datetime.today().strftime('%Y_%m_%d')}.csv", sep=";", index=False)

    def fit(self, X: pd.DataFrame, y=None):
        logger.info("FeatureEngineer: Fitting...")
        temp_X_for_fit = X.copy()
        
        # Prepare building_type before fitting
        if 'building_type' in temp_X_for_fit.columns:
            temp_X_for_fit['building_type'] = temp_X_for_fit['building_type'].apply(
                lambda x: x if x in self.top_building_types else 'other'
            )

        # Frequency encoding - learn mappings
        for col in self.frequency_encode_cols:
            if col in temp_X_for_fit.columns:
                freq_map = temp_X_for_fit[col].value_counts(normalize=True).to_dict()
                self.encodings_[col] = freq_map
                logger.info(f"FeatureEngineer: Learned frequency map for '{col}'.")

        # One hot encoding - learn categories
        for col in self.one_hot_encode_cols:
            if col in temp_X_for_fit.columns:
                self.categories_[col] = temp_X_for_fit[col].astype('category').cat.categories.tolist()
                logger.info(f"FeatureEngineer: Learned categories for '{col}': {self.categories_[col]}.")

        # Create features that will be used for scaling
        temp_X_transformed = self._create_features(temp_X_for_fit)
        
        # Determine which columns to scale
        self.columns_for_scaling_ = [col for col in self.columns_to_scale if col in temp_X_transformed.columns]
        if self.columns_for_scaling_:
            # Ensure all scaling columns are numeric
            for col in self.columns_for_scaling_:
                temp_X_transformed[col] = pd.to_numeric(temp_X_transformed[col], errors='coerce')
            
            self.scaler_.fit(temp_X_transformed[self.columns_for_scaling_])
            logger.info(f"FeatureEngineer: Fit scaler for columns: {self.columns_for_scaling_}")

        return self
    
    def _create_features(self, df):
        """Helper method to create engineered features"""
        df_copy = df.copy()
        
        # Feature creation from description
        feature_creation_map = {
            'elevator': (r'\bwinda\w*\b', 'elevator'),
            'balcony': (r'\b(balkon\w*|taras\w*)\b', 'balcony'),
            'garage': (r'\bgaraż\w*\b', 'garage'),
            'furnished': (r'\bmeble\w*\b', 'furnished')
        }

        if 'description' in df_copy.columns:
            for feature, (pattern, col_name) in feature_creation_map.items():
                if col_name not in df_copy.columns or df_copy[col_name].isnull().any():
                    has_feature = df_copy['description'].str.contains(pattern, case=False, na=False)
                    df_copy.loc[has_feature, col_name] = 1
                    
                    # Handle existing categorical values
                    if col_name in ['elevator', 'furnished']:
                        df_copy[col_name] = df_copy[col_name].replace({'Tak': 1, 'Nie': 0})
                    
                    df_copy[col_name] = df_copy[col_name].fillna(0).astype(int)
                    logger.info(f"FeatureEngineer: '{col_name}' feature engineered from description.")

        # Create rooms_per_area with proper numeric handling
        if "rooms" in df_copy.columns and "area" in df_copy.columns:
            # Ensure both columns are numeric
            df_copy['rooms'] = pd.to_numeric(df_copy['rooms'], errors='coerce')
            df_copy['area'] = pd.to_numeric(df_copy['area'], errors='coerce')
            
            # Create safe division avoiding division by zero
            safe_area = df_copy['area'].replace(0, np.nan)
            rooms_per_area = df_copy['rooms'] / safe_area
            
            # Handle infinite values and ensure float64 dtype
            rooms_per_area = rooms_per_area.replace([np.inf, -np.inf], np.nan)
            df_copy['rooms_per_area'] = rooms_per_area.fillna(0).astype('float64')
            
            logger.info("FeatureEngineer: 'rooms_per_area' feature created with proper numeric dtype.")

        return df_copy
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("FeatureEngineer: Transforming...")
        df_transformed = self._create_features(X)
        
        # Binary reduction
        for col, positive_val in self.binary_reduce_cols.items():
            if col in df_transformed.columns:
                df_transformed[col] = (df_transformed[col] == positive_val).astype(int)
                logger.info(f"FeatureEngineer: Reduced '{col}' to binary.")

        # Building type reduction
        if 'building_type' in df_transformed.columns:
            df_transformed['building_type'] = df_transformed['building_type'].apply(
                lambda x: x if x in self.top_building_types else 'other'
            )
            logger.info("FeatureEngineer: 'building_type' reduced.")
        
        # Frequency encoding
        for col in self.frequency_encode_cols:
            if col in df_transformed.columns and col in self.encodings_:
                df_transformed[f'{col}_freq'] = df_transformed[col].map(self.encodings_[col]).fillna(0).astype('float64')
                logger.info(f"FeatureEngineer: Applied frequency encoding for '{col}'.")

        # One hot encoding
        for col in self.one_hot_encode_cols:
            if col in df_transformed.columns and col in self.categories_:
                cat_type = pd.CategoricalDtype(categories=self.categories_[col], ordered=False)
                df_transformed[col] = df_transformed[col].astype(cat_type)
                dummies = pd.get_dummies(df_transformed[col], prefix=col, drop_first=self.drop_original_onehot, dtype=int)
                df_transformed = pd.concat([df_transformed, dummies], axis=1)
                
                if self.drop_original_onehot:
                    df_transformed.drop(columns=[col], inplace=True)
                logger.info(f"FeatureEngineer: Applied One-Hot Encoding for '{col}'.")

        # Scaling
        if self.columns_for_scaling_ and any(c in df_transformed.columns for c in self.columns_for_scaling_):
            # Ensure all columns to scale are numeric
            for col in self.columns_for_scaling_:
                if col in df_transformed.columns:
                    df_transformed[col] = pd.to_numeric(df_transformed[col], errors='coerce').fillna(0).astype('float64')
            
            cols_to_scale_present = [col for col in self.columns_for_scaling_ if col in df_transformed.columns]
            if cols_to_scale_present:
                df_transformed[cols_to_scale_present] = self.scaler_.transform(df_transformed[cols_to_scale_present])
                logger.info(f"FeatureEngineer: Scaled columns: {cols_to_scale_present}")

        # Drop final columns
        cols_to_actually_drop = [c for c in self.columns_to_drop_final if c in df_transformed.columns]
        if cols_to_actually_drop:
            df_transformed.drop(cols_to_actually_drop, axis=1, inplace=True)
            logger.info(f"FeatureEngineer: Dropped final columns: {cols_to_actually_drop}")

        # Final cleanup - ensure all remaining columns are numeric
        for col in df_transformed.columns:
            if df_transformed[col].dtype == 'object':
                logger.warning(f"Converting object column '{col}' to numeric")
                df_transformed[col] = pd.to_numeric(df_transformed[col], errors='coerce').fillna(0).astype('float64')

        df_transformed.reset_index(drop=True, inplace=True)
        logger.info("FeatureEngineer: Transformation complete.")

        return df_transformed