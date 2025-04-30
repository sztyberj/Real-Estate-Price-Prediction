import pandas as pd
import numpy as np
import sys
from pathlib import Path

class DataProcessor():
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def clean_price(self):
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
        self.df = self.df.dropna(subset=['price'])
    
        return self
    
    def clean_ppm(self):
        self.df.loc[:,'price_per_meter'] = self.df['price_per_meter'].fillna(self.df['price'] / self.df['area'])
        self.df['price_per_meter'] = pd.to_numeric(self.df['price_per_meter'], errors='coerce')

        return self

    def clean_rooms(self):
        self.df.loc[:, 'rooms'] = pd.to_numeric(self.df['rooms'], errors='coerce')
        self.df = self.df.dropna(subset=['rooms'])

        return self

    def clean_market_type(self):
        self.df = self.df.dropna(subset=['market_type'])

        return self

    def clean_furnished(self):
        self.df = self.df.dropna(subset=['furnished'])

        return self

    def clean_district(self):
        self.df = self.df.dropna(subset=['district'])

        return self

    def clean_building_type(self):
        self.df['building_type'] = self.df['building_type'].fillna('unknown')
        
        return self

    def clean_year_built(self):
        self.df.loc[self.df['year_built'] < 1300, 'year_built'] = np.nan
        self.df.loc[:,'year_built'] = self.df['year_built'].fillna(self.df['year_built'].median())
        
        return self

    def clean_rent(self):
        self.df.loc[:,'rent'] = self.df['rent'].fillna(self.df['rent'].median())

        return self

    def clean_finish_status(self):
        self.df = self.df.dropna(subset=['finish_status'])

        return self

    def clean_ownership(self):
        self.df = self.df.dropna(subset=['ownership'])

        return self

    def clean_heating(self):
        self.df.loc[:,'heating'] = self.df['heating'].fillna('unknown')

        return self

    def clean_elevator(self):
        self.df = self.df.dropna(subset=['elevator'])

        return self

    def clean_outliers(self, column, lower=0.05, upper=0.95):
        q_low = self.df[column].quantile(lower)
        q_high = self.df[column].quantile(upper)
        
        self.df.loc[:,column] = self.df[column].clip(lower=q_low, upper=q_high)

        return self

    def drop_duplicates(self):
        self.df = self.df.drop_duplicates(subset=['price', 'url'])
        self.df = self.df.reset_index(drop=True)

        return self

    def drop_columns(self, columns: [str]):
        self.df = self.df.drop(columns, axis=1)

        return self


if __name__ == "__main__":
    from src.eda.data_reader import DataReader
    
    DIR = Path().resolve()
    PROJECT_ROOT = DIR.parent
    sys.path.append(str(PROJECT_ROOT))

    reader = DataReader()
    df = reader.read()

    data_processor = (DataProcessor(df)
      .clean_price()
      .clean_ppm()
      .clean_rooms()
      .clean_market_type()
      .clean_furnished()
      .clean_district()
      .clean_building_type()
      .clean_year_built()
      .clean_rent()
      .clean_finish_status()
      .clean_ownership()
      .clean_heating()
      .clean_elevator()
      .clean_outliers('price')
      .clean_outliers('price_per_meter')
      .clean_outliers('rent')
      .drop_duplicates()
      .drop_columns(['source', 'date', 'url', 'title', 'ad_id', 'external_id'])
     )

    df = data_processor.df
    print(df)