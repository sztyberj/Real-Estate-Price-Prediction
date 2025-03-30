import pandas as pd
import numpy as np
import re
import os
import logging

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

pd.set_option('future.no_silent_downcasting', True)

class InitialCleaner:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def clean_price(self, df):
        self.logger.info("Cleaning price...")
        df['price'] = df['price'].str.replace(r"[^\d]", "", regex=True)
        df['price'] = df['price'].replace("", np.nan).astype("Int64")
        return df

    def clean_ppm(self, df):
        self.logger.info("Cleaning price per meter...")
        df['price_per_meter'] = df['price_per_meter'].str.replace(r"[^\d]", "", regex=True)
        return df

    def clean_floor(self, df):
        self.logger.info("Cleaning floor...")
        df['floor'] = df['floor'].str.split("/").str[0]
        df.loc[df['floor'].astype(str).str.lower().str.contains("parter", na=False), 'floor'] = 0
        return df

    def clean_furnished(self, df):
        self.logger.info("Cleaning furnished status...")
        df['furnished'] = df['furnished'].replace({"Tak": 1, "Nie": 0}).astype("Int64")
        return df

    def clean_address(self, df):
        self.logger.info("Cleaning address...")
        address_parts = df["district"].str.split(",", expand=True)
        address_parts = address_parts.map(lambda x: x.strip() if isinstance(x, str) else x)

        mask = address_parts.iloc[:, 2].str.lower() == "mazowieckie"
        address_parts.loc[mask, [4, 3, 2, 1]] = address_parts.loc[mask, [3, 2, 1, 0]].values

        del address_parts[4]
        address_parts.columns = ["street", "neighbourhood", "district", "voivodeship"]

        df = df.drop(columns=["district"])
        df = df.join(address_parts)
        return df

    def olx(self, df):
        self.logger.info("Starting initial cleaning pipeline for olx...")
        df = self.clean_price(df)
        df = self.clean_ppm(df)
        df = self.clean_floor(df)
        df = self.clean_furnished(df)
        self.logger.info("Initial cleaning completed.")
        return df

    def oto(self, df):
        self.logger.info("Starting initial cleaning pipeline for olx...")
        df = self.clean_price(df)
        df = self.clean_ppm(df)
        df = self.clean_floor(df)
        df = self.clean_furnished(df)
        df = self.clean_address(df)
        self.logger.info("Initial cleaning completed.")
        return df


if __name__ == "__main__":
    cleaner = InitialCleaner()

    # Test data
    df_test = pd.DataFrame({
        "price": ["1 200 000 zł", "999.000", ""],
        "price_per_meter": ["10 000 zł", "12.345", ""],
        "floor": ["3/10", "parter", "7"],
        "furnished": ["Tak", "Nie", None],
        "district": [
            "ul. Gratyny, Kępa Zawadowska, Wilanów, mazowieckie",
            "ul. Marszałkowska, Śródmieście, Warszawa, mazowieckie",
            "ul. Testowa, Prądnik, Kraków, małopolskie"
        ]
    })

    # OLX Test
    print("\n=== Test: OLX Cleaning ===")
    df_olx = cleaner.olx(df_test.copy())
    print(df_olx)

    # OTO Test
    print("\n=== Test: OTODOM Cleaning ===")
    df_oto = cleaner.oto(df_test.copy())
    print(df_oto)