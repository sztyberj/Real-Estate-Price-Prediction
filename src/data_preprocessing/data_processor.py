import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
import toml
import warnings
from src.utils.logging_config import logger

pd.set_option('display.float_format', lambda x: f'{x:.4f}')
pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings('ignore')

THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[2]
sys.path.append(str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data" / "cleaned"

with open(ROOT_DIR / "config.toml", 'r') as f:
    config = toml.load(f)

class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config = None):
        self.config = dict(config) if config is not None else {}

        dp_config = self.config.get('data_processing', {})

        self.columns_to_check = dp_config.get('columns_to_check', [])
        self.columns_to_clean = dp_config.get('columns_to_clean', [])
        self.columns_to_drop = dp_config.get('columns_to_drop', [])

        self.floor_map = dict(dp_config.get('floor_map', {}))

        self.lower = dp_config.get('outlier_lower', 0.05)
        self.upper = dp_config.get('outlier_upper', 0.95)
    
    @staticmethod
    def save_to_csv(df: pd.DataFrame , version: str):
        logger.info(f"DataProcessor: save to .csv as {version} version")
        df.to_csv(f"{DATA_DIR}/v{version}_{datetime.today().strftime('%Y_%m_%d')}.csv", sep=";", index=False)

    def fit(self, X, y=None):
        logger.info("Fit")
        temp_df = X.copy()

        #fit median year_built
        if 'year_built' in temp_df.columns:
            self.year_built_median = temp_df['year_built'].median()
            logger.info(f"DataProcessor: Learned median for 'year_built'.")
        else:
            self.year_built_median = None

        #fit median rent
        if 'rent' in temp_df.columns:
            temp_df['rent'] = pd.to_numeric(temp_df['rent'], errors='coerce')
            self.rent_median = temp_df['rent'].median()
            logger.info(f"DataProcessor: Learned median for 'rent'.")
        else:
            self.rent_median = None

        return self
        
    def transform(self, X, y=None):
        logger.info("Transform")
        df = X.copy()
        #price
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df.dropna(subset=['price'])
            logger.info("DataProcessor: 'price' transformed.")

        #ppm
        if 'price_per_meter' in df.columns:
            df.loc[:,'price_per_meter'] = df['price_per_meter'].fillna(df['price'] / df['area'])
            df['price_per_meter'] = pd.to_numeric(df['price_per_meter'], errors='coerce')
            logger.info("DataProcessor: 'ppm' transformed.")

        #rooms
        if 'rooms' in df.columns:
            df.loc[:, 'rooms'] = pd.to_numeric(df['rooms'], errors='coerce')
            df = df.dropna(subset=['rooms'])
            logger.info("DataProcessor: 'rooms' transformed.")

        #building_type
        if 'building_type' in df.columns:
            df['building_type'] = df['building_type'].fillna('unknown')
            logger.info("DataProcessor: 'building_type' filled.")

        #year_built
        if 'year_built' in df.columns:
            df.loc[df['year_built'] < 1300, 'year_built'] = np.nan
            df.loc[:,'year_built'] = df['year_built'].fillna(self.year_built_median)
            logger.info("DataProcessor: 'year_built' filled.")
        
        #rent
        if 'rent' in df.columns:
            df.loc[:,'rent'] = df['rent'].fillna(self.rent_median)
            logger.info("DataProcessor: 'rent' filled.")

        #drop_na
        existing_columns_to_check = [col for col in self.columns_to_check if col in df.columns]
        if existing_columns_to_check:
            df = df.dropna(subset=existing_columns_to_check)
            logger.info(f"DataProcessor: 'na' i columns: {existing_columns_to_check}  dropped.")

        #heating
        if 'heating' in df.columns:
            df.loc[:,'heating'] = df['heating'].fillna('unknown')
            logger.info("DataProcessor: 'heating' filled.")

        #duplicates
        required_columns = {'price', 'url'}
        if required_columns.issubset(df.columns):
            df = df.drop_duplicates(subset=['price', 'url'])
            df = df.reset_index(drop=True)
            logger.info("DataProcessor: duplicates dropped.")

        #clean_outliers
        existing_columns_to_clean = [col for col in self.columns_to_clean if col in df.columns]
        if existing_columns_to_clean:
            for i in existing_columns_to_clean:
                q_low = df[i].quantile(self.lower)
                q_high = df[i].quantile(self.upper)
        
                df.loc[:,i] = df[i].clip(lower=q_low, upper=q_high)
                logger.info(f"DataProcessor: {i} outliers cleaned.")

        #drop_columns
        existing_columns_to_drop = [col for col in self.columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            df = df.drop(columns=existing_columns_to_drop, errors='ignore')
            logger.info(f"DataProcessor: columns {existing_columns_to_drop} dropped.")

        if 'floor' in df.columns:
            if not df['floor'].isnull().all():
                df[['floor', 'building_max_floor']] = df['floor'].astype(str).str.split('/', expand=True)

            df['floor'] = df['floor'].replace(self.floor_map)
            df['building_max_floor'] = pd.to_numeric(df['building_max_floor'], errors='coerce')

            df.loc[df['floor'] == 'poddasze', 'floor'] = df['building_max_floor'] + 1

            df['is_above_10_floor'] = df['floor'].astype(str).str.contains('>').astype(int)

            df['floor'] = df['floor'].astype(str).str.replace('>', '', regex=False)
            df['floor'] = pd.to_numeric(df['floor'], errors='coerce')

            df.loc[df['building_max_floor'] > 60, 'building_max_floor'] = np.nan

            if 'floor' in df.columns and 'building_max_floor' in df.columns:
                df = df.dropna(subset=['floor', 'building_max_floor'])

            logger.info("DataProcessor: 'floor' and 'building_max_floor' processed.")
        
        return df

if __name__ == "__main__":
    #Run from ./Real Estate Price Prediction/
    from src.eda.data_reader import DataReader
    reader = DataReader()
    df = reader.read()

    data_processor = DataProcessor(config)
    data_processor.fit(df)
    df = data_processor.transform(df)
    data_processor.save_to_csv(df, version="T")

    print(df)