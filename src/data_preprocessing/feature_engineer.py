import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
import toml
import warnings

pd.set_option('display.float_format', lambda x: f'{x:.4f}')
pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings('ignore')

THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[2]
sys.path.append(str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data" / "processed"

with open(ROOT_DIR / "config.toml", 'r') as f:
    config = toml.load(f)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, config_feature = None):
        self.config = config_feature if config_feature is not None else {}

        fe_config = self.config.get('feature_engineering', {})

        self.freq_encode_cols = fe_config.get('frequency_encode_cols', [])
        self.encodings = {}

        self.one_hot_encode_cols = fe_config.get('one_hot_encode_cols', [])
        self.drop_original_onehot = fe_config.get('drop_original_onehot', True)
        self.categories_ = {}
        
        self.luxury_quantile = fe_config.get('luxury_quantile', 0.9)
        self.price_per_meter_quantile_threshold = None

        self.binary_reduce_cols = fe_config.get('binary_reduce_cols', {})

        self.cols_to_drop_final = fe_config.get('columns_to_drop_final', [])

        self.floor_map = fe_config.get('floor_map', {})

        self.top_building_types = fe_config.get('top_building_types', [])

    @staticmethod
    def save_to_csv(df: pd.DataFrame , version: str):
        print(f"FeatureEngineer: save to .csv as {version} version")
        df.to_csv(f"{DATA_DIR}/v{version}_{datetime.today().strftime('%Y_%m_%d')}.csv", sep=";", index=False)
    
    def fit(self, X: pd.DataFrame, y=None):
        temp_X_for_fit = X.copy()
        
        #Preprocessing
        if 'building_type' in self.one_hot_encode_cols and 'building_type' in temp_X_for_fit.columns:
            temp_X_for_fit['building_type'] = temp_X_for_fit['building_type'].apply(
                lambda x: x if x in self.top_building_types else 'other'
            )
            print("FeatureEngineer: 'building_type' reduced for fit purposes (to learn correct OHE categories).")

        #Frequency encoding
        for col in self.freq_encode_cols:
            if col in X.columns:
                freq_map = X[col].value_counts(normalize=True).to_dict()
                self.encodings[col] = freq_map
                print(f"FeatureEngineer: Learned frequency map for '{col}'.")

        #One hot encoding
        for col in self.one_hot_encode_cols:
            if col in temp_X_for_fit.columns:
                self.categories_[col] = pd.Categorical(temp_X_for_fit[col]).categories.tolist()
                print(f"FeatureEngineer: Learned categories for '{col}': {self.categories_[col]}.")

        #Is_luxury threshold
        if 'price_per_meter' in X.columns:
            self.price_per_meter_quantile_threshold = X['price_per_meter'].quantile(self.luxury_quantile)
            print(f"FeatureEngineer: Learned luxury threshold for 'price_per_meter': {self.price_per_meter_quantile_threshold:.2f}")
        else:
            self.price_per_meter_quantile_threshold = 0

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_transformed = X.copy()

        #--- static columns
        #1. process floor
        if 'floor' in df_transformed.columns:
            if not df_transformed['floor'].isnull().all():
                df_transformed[['floor', 'building_max_floor']] = df_transformed['floor'].astype(str).str.split('/', expand=True)

            df_transformed['floor'] = df_transformed['floor'].replace(self.floor_map)
            df_transformed['building_max_floor'] = pd.to_numeric(df_transformed['building_max_floor'], errors='coerce')

            df_transformed.loc[df_transformed['floor'] == 'poddasze', 'floor'] = df_transformed['building_max_floor'] + 1

            df_transformed['is_above_10_floor'] = df_transformed['floor'].astype(str).str.contains('>').astype(int)

            df_transformed['floor'] = df_transformed['floor'].astype(str).str.replace('>', '', regex=False)
            df_transformed['floor'] = pd.to_numeric(df_transformed['floor'], errors='coerce')

            df_transformed.loc[df_transformed['building_max_floor'] > 60, 'building_max_floor'] = np.nan

            if 'floor' in df_transformed.columns and 'building_max_floor' in df_transformed.columns:
                df_transformed = df_transformed.dropna(subset=['floor', 'building_max_floor'])

            print("FeatureEngineer: 'floor' and 'building_max_floor' processed.")

        #2. has_elevator_in_desc
        if 'description' in df_transformed.columns:
            has_elevator = df_transformed['description'].str.contains(r'\bwinda\w*\b', case=False, na=False)
            df_transformed.loc[has_elevator, 'elevator'] = 1
            df_transformed['elevator'] = df_transformed['elevator'].replace({'Tak': 1, 'Nie': 0})
            df_transformed['elevator'] = df_transformed['elevator'].fillna(0).astype(int)
            print("FeatureEngineer: 'elevator' feature created.")

        #3. has_balcony_in_desc
        if 'description' in df_transformed.columns:
            has_balcony = df_transformed['description'].str.contains(r'\b(balkon\w*|taras\w*)\b', case=False, na=False)
            df_transformed.loc[has_balcony, 'balcony'] = 1
            df_transformed['balcony'] = df_transformed['balcony'].fillna(0).astype(int) # Change to int
            print("FeatureEngineer: 'balcony' feature created.")

        # 4. has_garage_in_desc
        if 'description' in df_transformed.columns:
            has_garage = df_transformed['description'].str.contains(r'\bgaraż\w*\b', case=False, na=False)
            df_transformed.loc[has_garage, 'garage'] = 1
            df_transformed['garage'] = df_transformed['garage'].fillna(0).astype(int) # Change to int
            print("FeatureEngineer: 'garage' feature created.")

        # 5. has_furniture_in_desc
        if 'description' in df_transformed.columns:
            has_furniture = df_transformed['description'].str.contains(r'\bmeble\w*\b', case=False, na=False)
            df_transformed.loc[has_furniture, 'furnished'] = 1
            df_transformed['furnished'] = df_transformed['furnished'].replace({'Tak': 1, 'Nie': 0})
            df_transformed['furnished'] = df_transformed['furnished'].fillna(0).astype(int)
            print("FeatureEngineer: 'furnished' feature created.")

        # 6. create room_per_area col
        if "rooms" in df_transformed.columns and "area" in df_transformed.columns and not df_transformed['area'].eq(0).any():
            df_transformed["rooms_per_area"] = (df_transformed["rooms"] / df_transformed["area"]).astype(float)
            # Handle potential division by zero results
            df_transformed['rooms_per_area'].replace([np.inf, -np.inf], np.nan, inplace=True)
            df_transformed['rooms_per_area'].fillna(0, inplace=True) # Fill with 0 or median/mean
            print("FeatureEngineer: 'rooms_per_area' feature created.")

        # 7. reduce_to_binary
        for col, positive_val in self.binary_reduce_cols.items():
            if col in df_transformed.columns:
                df_transformed[col] = (df_transformed[col] == positive_val).astype(int)
                print(f"FeatureEngineer: Reduced '{col}' to binary.")

        # 8. reduce_building_type
        if 'building_type' in df_transformed.columns:
            top_types = self.top_building_types
            df_transformed['building_type'] = df_transformed['building_type'].apply(
                lambda x: x if x in top_types else 'other'
            )
            print("FeatureEngineer: 'building_type' reduced.")

        #required 'fit'
        # 9. create_luxury_col (wymaga price_per_meter i nauczonego progu)
        if "price_per_meter" in df_transformed.columns and self.price_per_meter_quantile_threshold is not None:
            df_transformed["is_luxury"] = (df_transformed["price_per_meter"] > self.price_per_meter_quantile_threshold).astype(int)
            print("FeatureEngineer: 'is_luxury' feature created.")

        #10. frequency_encoding
        for col in self.freq_encode_cols:
            if col in df_transformed.columns and col in self.encodings:
                df_transformed[f'{col}_freq'] = df_transformed[col].map(self.encodings[col]).fillna(0)
                print(f"FeatureEngineer: Applied frequency encoding for '{col}'.")

        #11. one_hot_encoding
        for col in self.one_hot_encode_cols:
            if col in df_transformed.columns and col in self.categories_:
                df_transformed[col] = pd.Categorical(df_transformed[col], categories=self.categories_[col])
                
                dummies = pd.get_dummies(df_transformed[col], prefix=col, 
                                         drop_first=self.drop_original_onehot, dtype=int)
                df_transformed = pd.concat([df_transformed, dummies], axis=1)
                
                if self.drop_original_onehot:
                    df_transformed.drop(columns=[col], inplace=True)
                print(f"FeatureEngineer: Applied One-Hot Encoding for '{col}'.")

        #12. drop columns
        cols_to_actually_drop = [c for c in self.cols_to_drop_final if c in df_transformed.columns]
        if cols_to_actually_drop:
            df_transformed = df_transformed.drop(cols_to_actually_drop, axis=1)
            print(f"FeatureEngineer: Dropped columns: {cols_to_actually_drop}")

        df_transformed = df_transformed.reset_index(drop=True)

        print("FeatureEngineer: Transformation complete.")
        return df_transformed

if __name__ == "__main__":
    from src.eda.data_reader import DataReader
    from src.data_preprocessing.data_processor import DataProcessor
    
    reader = DataReader()
    df = reader.read()
    
    data_processor = DataProcessor(config)
    data_processor.fit(df)
    df = data_processor.transform(df)
    data_processor.save_to_csv(df, version="T")

    X = df.drop('price', axis=1)
    y = df['price']

    engineer = FeatureEngineer(config)
    engineer.fit(X)
    X = engineer.transform(X)
    print(X)