import re
import os
import sys
import logging
import pandas as pd
sys.path.append("../data/cleanup")

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class DataReader:
    def __init__(self, input_dir = "../data/cleanup/"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.input_dir = input_dir

    def read(self):
        self.frames_olx = []
        self.frames_oto = []
        for filename in os.listdir(self.input_dir):
            if "olx" in filename and filename.endswith(".csv"):
                tmp = pd.read_csv(os.path.join(self.input_dir, filename))
                self.frames_olx.append(self.tmp)
            if "oto" in filename and filename.endswith(".csv"):
                tmp = pd.read_csv(os.path.join(self.input_dir, filename))
                self.frames_oto.append(self.tmp)
        self.pd_olx = pd.concat(self.frames_olx, ignore_index=True) if self.frames_olx else pd.DataFrame()
        self.pd_oto = pd.concat(self.frames_oto, ignore_index=True) if self.frames_oto else pd.DataFrame()

        return self.pd_olx, self.pd_oto
    


if __name__ == "__main__":
    reader = DataReader()

    olx, oto = reader.read()

    print(olx.head(5))