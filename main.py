from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from typing import List, Optional
from pathlib import Path
import uvicorn
from src.utils.logging_config import logger
import toml
import sys
import json
import joblib
import numpy as np

THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[0]
sys.path.append(str(ROOT_DIR))

with open(ROOT_DIR / "config.toml", 'r') as f:
    config = toml.load(f)
    config = json.loads(json.dumps(config))

api_config = config.get('model_params', {})
version = api_config.get('version', "")
title = api_config.get('title', "Real Estate Price Prediction API")
host = api_config.get('host', "127.0.0.1")
port = api_config.get('port', 8000)

MODEL_COMPONENTS_PATH = ROOT_DIR / f"models/real_estate_model_components_{version}.joblib"

try:
    logger.info(f"[TRY] Load model components from {MODEL_COMPONENTS_PATH}")
    model_components = joblib.load(MODEL_COMPONENTS_PATH)
except FileNotFoundError:
    model_components = None
    logger.error(f"Error: Can't find model components at {MODEL_COMPONENTS_PATH}")

#BaseModel Data
class RawDataPoint(BaseModel):
    area: float
    rooms: int
    floor: int
    building_max_floor: Optional[int] = None
    year_built: Optional[int] = None
    district: Optional[str] = None
    market_type: Optional[str] = None
    building_type: Optional[str] = None
    finish_status: Optional[str] = None
    ownership: Optional[str] = None
    heating: Optional[str] = None
    garage: Optional[bool] = Field(None)
    balcony: Optional[bool] = Field(None)
    furnished: Optional[bool] = Field(None)
    elevator: Optional[bool] = Field(None)
    description: Optional[str] = Field("")
    price_per_meter: Optional[float] = Field(0.0)
    rent: Optional[int] = Field(0)
    is_above_10_floor: Optional[int] = Field(0)

app = FastAPI(title=title)

@app.post("/predict/")
def predict(raw_data_points: List[RawDataPoint]):
    if model_components is None:
        logger.error("Model components are not loaded.")
        return {"error": "Model is not available. Check server logs."}

    input_df = pd.DataFrame([item.model_dump() for item in raw_data_points])

    #Prepare data
    bool_cols = ['garage', 'balcony', 'furnished', 'elevator']
    for col in bool_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].apply(lambda x: 1 if x is True else 0)

    temp_floor_col = []
    for _, row in input_df.iterrows():
        floor, max_floor = row.get('floor'), row.get('building_max_floor')
        if pd.notna(max_floor):
            temp_floor_col.append(f"{int(floor)}/{int(max_floor)}")
        else:
            temp_floor_col.append(str(int(floor)))
    input_df['floor'] = temp_floor_col

    if 'price' not in input_df.columns:
        input_df['price'] = 0

    logger.info("Step 1: Applying data_processor")
    processed_df = model_components['data_processor'].transform(input_df)

    if 'price' in processed_df.columns:
        processed_df = processed_df.drop(columns=['price'])

    logger.info("Step 2: Applying feature_engineer")
    pipeline = model_components['feature_model_pipeline']
    engineered_df = pipeline.named_steps['feature_engineering'].transform(processed_df)

    logger.info("Ensuring correct feature order for the model")
    model_features = pipeline.named_steps['model'].get_booster().feature_names
    engineered_df = engineered_df[model_features]

    logger.info("Step 3: Predicting with the model")
    prediction_scaled = pipeline.named_steps['model'].predict(engineered_df)

    #Predict
    logger.info("Step 4: Inverse transforming the scaled prediction")
    prediction = model_components['y_scaler'].inverse_transform(prediction_scaled.reshape(-1, 1))

    logger.info(f"Prediction successful. Result: {prediction.flatten().tolist()}")
    return {"predictions": prediction.flatten().tolist()}

@app.get("/")
def root():
    if model_components:
        return {"message": f"Model components from '{MODEL_COMPONENTS_PATH.name}' are ready to use."}
    else:
        return {"error": "Model components are not available. Check server logs."}
    
if __name__ == "__main__":
    logger.info("--- Running a simple prediction test ---")
    
    if model_components:
        test_data = [
            {   
                "area": 100.0,
                "rooms": 5,
                "floor": 2,
                "building_max_floor": 10,
                "year_built": 2022,
                "district": "Wilanów",
                "market_type": "pierwotny",
                "building_type": "apartamentowiec",
                "finish_status": "do wykończenia",
                "ownership": "pełna własność",
                "heating": "miejskie",
                "garage": True,
                "balcony": True,
                "elevator": False,
                "furnished": False,
            }
        ]
        
        test_points = [RawDataPoint(**data) for data in test_data]
        test_predictions = predict(test_points)
        
        print("\n--- Test Result ---")
        print("Input data:")
        print(json.dumps(test_data, indent=2, ensure_ascii=False))
        print("\nPrediction result:")
        print(test_predictions)
        print("-------------------\n")
    else:
        logger.error("Test cannot be performed because model components were not loaded.")

    logger.info(f"Starting FastAPI server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)