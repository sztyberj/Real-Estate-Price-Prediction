import re
import os
import sys
import logging
import pandas as pd
from pathlib import Path
from src.utils.logging_config import logger

THIS_FILE = Path(__file__).resolve()
ROOT_DIR  = THIS_FILE.parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"

class DataReader:
    def __init__(self, input_dir: str = DATA_DIR, ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.input_dir = input_dir

    def read(self) -> pd.DataFrame:
        self.frames_df = []
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".csv"):
                tmp = pd.read_csv(os.path.join(self.input_dir, filename), sep=";")
                self.frames_df.append(tmp)

        self.df = pd.concat(self.frames_df, ignore_index=True) if self.frames_df else pd.DataFrame()
        logger.info("DataReader: Readed files.")
        return self.df
    

if __name__ == "__main__":
    reader = DataReader()

    df = reader.read()
    print(df.info())
    print(df.head(5))
	
	
	