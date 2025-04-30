import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

pd.set_option('display.float_format', lambda x: f'{x:.4f}')

class FeatureEngineer():
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.scalers = {}
        self.encodings = {}

    @staticmethod
    def load(path: str):
        print(f"Loading PreProcessor from: {path}")
        
        return joblib.load(path)

    def save(self, path: str):
        joblib.dump(self, path)
        print(f"PreProcessor saved to: {path}")

    def process_floor(self):
        #split floor column by sep "/"
        self.df[['floor', 'building_max_floor']]  = self.df['floor'].str.split('/', expand=True)
        
        floor_map = {
        'suterena': -1,
        'parter': 0,
        }
    
        self.df['floor'] = self.df['floor'].replace(floor_map)
    
        self.df['building_max_floor'] = pd.to_numeric(self.df['building_max_floor'], errors='coerce')
        self.df.loc[df['floor'] == 'poddasze', 'floor'] = self.df['building_max_floor'] + 1
            
        self.df['is_above_10_floor'] = self.df['floor'].astype(str).str.contains('>').astype(int)
    
        self.df.loc[:,'floor'] = self.df['floor'].astype(str).str.replace('>', '', regex=False)
    
        self.df['floor'] = pd.to_numeric(self.df['floor'], errors='coerce')

        self.df.loc[self.df['building_max_floor'] > 60, 'building_max_floor'] = np.nan
        
        self.df = self.df.dropna(subset=['floor', 'building_max_floor'])
    
        return self

    def has_elevator_in_desc(self):
        has_elevator = self.df['description'].str.contains(r'\bwinda\w*\b', case=False, na=False)
        self.df.loc[has_elevator, 'elevator'] = 1

        self.df['elevator'] = self.df['elevator'].replace({'Tak': 1, 'Nie': 0})
        self.df['elevator'] = self.df['elevator'].fillna(0).astype(int)
    
        return self

    def has_balcony_in_desc(self):
        has_balcony = self.df['description'].str.contains(r'\b(balkon\w*|taras\w*)\b', case=False, na=False)
        self.df.loc[has_balcony, 'balcony'] = 1

        self.df['balcony'] = self.df['balcony'].fillna(0)
    
        return self


    def has_garage_in_desc(self):
        has_garage = self.df['description'].str.contains(r'\bgaraż\w*\b', case=False, na=False)
        self.df.loc[has_garage, 'garage'] = 1
        self.df['garage'] = self.df['garage'].fillna(0)
    
        return self

    def has_furniture_in_desc(self):
        has_furniture = self.df['description'].str.contains( r'\bmeble\w*\b', case=False, na=False)
        self.df.loc[has_furniture, 'furnished'] = 1

        self.df['furnished'] = self.df['furnished'].replace({'Tak': 1, 'Nie': 0})
        self.df['furnished'] = self.df['furnished'].fillna(0).astype(int)
    
        return self
        
    def create_luxury_col(self):
        self.df["is_luxury"] = (self.df["price_per_meter"] > self.df["price_per_meter"].quantile(0.90)).astype(int)

        return self

    def create_room_per_area_col(self):
        self.df["rooms_per_area"] = (self.df["rooms"] / self.df["area"]).astype(float)

        return self

    def reduce_to_binary(self, column: str, positive_value: str, new_name: str = None):
        if new_name is None:
            new_name = column
        self.df[new_name] = (self.df[column] == positive_value).astype(int)
        
        return self

    def frequency_encoding(self, column: str):
        freq_map = self.df[column].value_counts(normalize=True)

        self.encodings[column] = freq_map.to_dict()
        
        self.df[f'{column}_freq'] = self.df[column].map(freq_map)
        
        return self

    def one_hot_encode(self, column: str, drop_original: bool = True):
        dummies = pd.get_dummies(self.df[column], prefix=column, drop_first=True, dtype=int)
        self.df = pd.concat([self.df, dummies], axis=1)
        if drop_original:
            self.df.drop(columns=[column], inplace=True)
        return self

    def reduce_building_type(self):
        top_types = {'apartment', 'block', 'tenement'}
        self.df['building_type'] = self.df['building_type'].apply(
            lambda x: x if x in top_types else 'other'
        )
        return self

    def drop_columns(self, columns: [str]):
        self.df = self.df.drop(columns, axis=1)
        self.df = self.df.reset_index(drop=True)
        
        return self

    def scale_columns(self, columns: list, method='standard'):
        for col in columns:
            if col not in self.df.columns:
                print(f"[SKIP] Column '{col}' not found.")
                continue

            scaler = StandardScaler()
            self.df[col] = scaler.fit_transform(self.df[[col]])
            self.scalers[col] = scaler

        print(f"Scaled columns: {columns}")
        return self

if __name__ == "__main__":
    DIR = Path().resolve()
    PROJECT_ROOT = DIR.parent
    sys.path.append(str(PROJECT_ROOT))

    from src.eda.data_reader import DataReader
    from src.data_preprocessing import DataProcessor
    
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
    
    feature_engineer = (FeatureEngineer(df)
        .process_floor()
        .has_elevator_in_desc()
        .has_balcony_in_desc()
        .has_garage_in_desc()
        .has_furniture_in_desc()
        .create_luxury_col()
        .create_room_per_area_col()
        .reduce_to_binary('heating' , 'miejskie', 'district_heating')
        .reduce_to_binary('ownership' , 'full_ownership', 'full_ownership')
        .reduce_to_binary('market_type' , 'pierwotny', 'primary_market')
        .reduce_building_type()
        .frequency_encoding('district')
        .one_hot_encode('finish_status')
        .one_hot_encode('building_type')
        .drop_columns(['description', 'ownership', 'heating', 'market_type', 'district'])
        .scale_columns(['price', 'price_per_meter', 'area', 'year_built', 'rent', 'rooms', 'floor', 'building_max_floor'])
         )

    df = feature_engineer.df

    print(df)