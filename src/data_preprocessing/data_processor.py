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

DATA_DIR = ROOT_DIR / "data" / "cleaned"

with open(ROOT_DIR / "config.toml", 'r') as f:
    config = toml.load(f)


class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config = None):
        self.config = config if config is not None else {}

        dp_config = self.config.get('data_processing', {})

        self.columns_to_check = dp_config.get('columns_to_check', [])
        self.columns_to_clean = dp_config.get('columns_to_clean', [])
        self.columns_to_drop = dp_config.get('columns_to_drop', [])

        self.lower = dp_config.get('outlier_lower', 0.05)
        self.upper = dp_config.get('outlier_upper', 0.95)

    @staticmethod
    def save_to_csv(df: pd.DataFrame , version: str):
        print(f"DataProcessor: save to .csv as {version} version")
        df.to_csv(f"{DATA_DIR}/v{version}_{datetime.today().strftime('%Y_%m_%d')}.csv", sep=";", index=False)

    def fit(self, X, y=None):
        temp_df = X.copy()

        #fit median year_built
        if 'year_built' in temp_df.columns:
            self.year_built_median = temp_df['year_built'].median()
            print(f"DataProcessor: Learned median for 'year_built'.")
        else:
            self.year_built_median = None

        #fit median rent
        if 'rent' in temp_df.columns:
            temp_df['rent'] = pd.to_numeric(temp_df['rent'], errors='coerce')
            self.rent_median = temp_df['rent'].median()
            print(f"DataProcessor: Learned median for 'rent'.")
        else:
            self.rent_median = None

        return self
        
    def transform(self, X, y=None):
        df = X.copy()
        #price
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df.dropna(subset=['price'])
            print("DataProcessor: 'price' transformed.")

        #ppm
        if 'price_per_meter' in df.columns:
            df.loc[:,'price_per_meter'] = df['price_per_meter'].fillna(df['price'] / df['area'])
            df['price_per_meter'] = pd.to_numeric(df['price_per_meter'], errors='coerce')
            print("DataProcessor: 'ppm' transformed.")

        #rooms
        if 'rooms' in df.columns:
            df.loc[:, 'rooms'] = pd.to_numeric(df['rooms'], errors='coerce')
            df = df.dropna(subset=['rooms'])
            print("DataProcessor: 'rooms' transformed.")

        #building_type
        if 'building_type' in df.columns:
            df['building_type'] = df['building_type'].fillna('unknown')
            print("DataProcessor: 'building_type' filled.")

        #year_built
        if 'year_built' in df.columns:
            df.loc[df['year_built'] < 1300, 'year_built'] = np.nan
            df.loc[:,'year_built'] = df['year_built'].fillna(self.year_built_median)
            print("DataProcessor: 'year_built' filled.")
        
        #rent
        if 'rent' in df.columns:
            df.loc[:,'rent'] = df['rent'].fillna(self.rent_median)
            print("DataProcessor: 'rent' filled.")

        #drop_na
        existing_columns_to_check = [col for col in self.columns_to_check if col in df.columns]
        if existing_columns_to_check:
            df = df.dropna(subset=existing_columns_to_check)
            print(f"DataProcessor: 'na' i columns: {existing_columns_to_check}  dropped.")

        #heating
        if 'heating' in df.columns:
            df.loc[:,'heating'] = df['heating'].fillna('unknown')
            print("DataProcessor: 'heating' filled.")

        #duplicates
        required_columns = {'price', 'url'}
        if required_columns.issubset(df.columns):
            df = df.drop_duplicates(subset=['price', 'url'])
            df = df.reset_index(drop=True)
            print("DataProcessor: duplicates dropped.")

        #clean_outliers
        existing_columns_to_clean = [col for col in self.columns_to_clean if col in df.columns]
        if existing_columns_to_clean:
            for i in existing_columns_to_clean:
                q_low = df[i].quantile(self.lower)
                q_high = df[i].quantile(self.upper)
        
                df.loc[:,i] = df[i].clip(lower=q_low, upper=q_high)
                print(f"DataProcessor: {i} outliers cleaned.")

        #drop_columns
        existing_columns_to_drop = [col for col in self.columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            df = df.drop(columns=existing_columns_to_drop, errors='ignore')
            print(f"DataProcessor: columns {existing_columns_to_drop} dropped.")
        
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