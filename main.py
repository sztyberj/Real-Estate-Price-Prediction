from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from typing import List, Optional
import uvicorn
from src.utils.logging_config import logger
import toml
import json
import joblib

with open("config.toml", 'r') as f:
    config = toml.load(f)
    config = json.loads(json.dumps(config))

api_config = config.get('api', {})
pipeline_params = config.get('pipeline', {})
version = pipeline_params['version']
print(f'Pipeline version {version}')
title = api_config.get('title', "Real Estate Price Prediction API")
host = api_config.get('host')
port = api_config.get('port')

try:
    logger.info(f"[TRY] Load model components")
    loaded_X_pipeline = joblib.load(f'models/X_pipeline_{version}.joblib')
    loaded_model = joblib.load(f'models/model_{version}.joblib')
    logger.info("Successfully loaded all model components.")

except FileNotFoundError:
    logger.error(f"Error: Can't find model components")

#BaseModel Data
class RawDataPoint(BaseModel):
    area: float
    rooms: int
    floor: int
    building_max_floor: Optional[int] = None
    year_built: Optional[int] = None
    district: Optional[str] = None
    market_type: Optional[str] = None
    furnished: Optional[bool] = None
    elevator: Optional[bool] = None
    building_type: Optional[str] = None
    finish_status: Optional[str] = None
    ownership: Optional[str] = None
    heating: Optional[str] = None
    garage: Optional[bool] = Field(None)
    rent: Optional[int] = Field(0)
    is_above_10_floor: Optional[int] = Field(0)
    
def make_prediction(raw_data_df):    
    processed_X = loaded_X_pipeline.transform(raw_data_df)
    prediction_scaled = loaded_model.predict(processed_X)
    
    return prediction_scaled.ravel()

app = FastAPI(title=title)

@app.post("/predict/")
def predict(raw_data_points: List[RawDataPoint]):
    if loaded_model is None or loaded_X_pipeline is None:
        logger.error("Model components are not loaded.")
        return {"error": "Model is not available. Check server logs."}

    input_df = pd.DataFrame([item.model_dump() for item in raw_data_points])

    prediction = make_prediction(input_df)

    logger.info(f"Prediction successful. Result: {prediction.flatten().tolist()}")
    return {"predictions": prediction.flatten().tolist()}

@app.get("/")
def root():
    if loaded_model and loaded_X_pipeline:
        return {"message": f"Model components are ready to use."}
    else:
        return {"error": "Model components are not available. Check server logs."}
    
if __name__ == "__main__":
    logger.info("--- Running a simple prediction test ---")
    
    if loaded_model or loaded_X_pipeline:
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
                "balcony": False,
                "elevator": True,
                "furnished": True,
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