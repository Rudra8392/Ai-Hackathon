from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import joblib
import uvicorn
from tensorflow import load_model
import time

app = FastAPI(title="Oil & Gas Production AI API")

# Data models
class ProductionData(BaseModel):
    timestamp: datetime
    total_flow: float
    oil_flow: float
    water_flow: float
    gas_flow: float
    pressure: float
    temperature: float
    gor: float

class PredictionResponse(BaseModel):
    timestamp: datetime
    predicted_flow: float
    predicted_pressure: float
    predicted_gor: float
    confidence: float
    prediction_time: float

# Load pre-trained models
try:
    flow_model = load_model('models/flow_prediction_model')
    scaler = joblib.load('models/scaler.joblib')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    flow_model = None
    scaler = None

@app.get("/")
async def root():
    return {"message": "Oil & Gas Production AI API"}

@app.post("/predict/flow", response_model=PredictionResponse)
async def predict_flow(data: ProductionData):
    try:
        # Prepare input data
        input_data = np.array([[
            data.total_flow,
            data.oil_flow,
            data.water_flow,
            data.gas_flow,
            data.pressure,
            data.temperature,
            data.gor
        ]])
        
        # Scale input data
        scaled_input = scaler.transform(input_data)
        
        # Make prediction
        start_time = time.time()
        if flow_model is not None:
            prediction = flow_model.predict(scaled_input.reshape(1, 1, -1))
            predicted_values = scaler.inverse_transform(prediction)[0]
            predicted_flow, predicted_pressure, predicted_gor = predicted_values[0], predicted_values[4], predicted_values[6]
        else:
            # Dummy prediction
            predicted_flow = data.total_flow * (1 + np.random.normal(0, 0.1))
            predicted_pressure = data.pressure * (1 + np.random.normal(0, 0.05))
            predicted_gor = data.gor * (1 + np.random.normal(0, 0.1))
        
        end_time = time.time()
        prediction_time = end_time - start_time
            
        return PredictionResponse(
            timestamp=data.timestamp,
            predicted_flow=float(predicted_flow),
            predicted_pressure=float(predicted_pressure),
            predicted_gor=float(predicted_gor),
            confidence=0.95,
            prediction_time=prediction_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    print("Starting Oil & Gas Production AI API")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
