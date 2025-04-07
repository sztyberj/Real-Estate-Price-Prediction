import pandas as pd
import numpy as np
import os
import logging
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from initial_cleaner import InitialCleaner

def clean_files():
    input_dir = "../../data/raw/"
    output_dir = "../../data/cleanup/"

    os.makedirs(output_dir, exist_ok=True)

    cleaner = InitialCleaner()

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path):
                logging.info(f"Skipping file (already exists): {filename}")
                continue

            logging.info(f"Processing file: {filename}")
            try:
                df = pd.read_csv(input_path, sep=";")

                if "OLX" in filename.upper():
                    df_cleaned = cleaner.olx(df)
                elif "OTO" in filename.upper():
                    df_cleaned = cleaner.oto(df)
                else:
                    logging.warning(f"Skipped file (unrecognized name): {filename}")
                    continue

                df_cleaned.to_csv(output_path, index=False)
                logging.info(f"Saved cleaned file to: {output_path}")

            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")

def main():
    clean_files()

if __name__ == '__main__':
    main()