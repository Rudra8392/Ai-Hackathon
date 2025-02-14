from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import joblib
import uvicorn

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


class PredictionResponse(BaseModel):
    timestamp: datetime
    predicted_flow: float
    predicted_pressure: float
    confidence: float


# Load pre-trained models (in practice, these would be your actual trained models)
try:
    flow_model = joblib.load('models/flow_prediction_model.joblib')
    decline_model = joblib.load('models/decline_detection_model.joblib')
except:
    print("Warning: Models not found. Using dummy models for demonstration.")
    flow_model = None
    decline_model = None


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
            data.temperature
        ]])

        # Make prediction (using dummy values for demonstration)
        if flow_model is not None:
            prediction = flow_model.predict(input_data)
            predicted_flow = float(prediction[0])
        else:
            # Dummy prediction
            predicted_flow = data.total_flow * (1 + np.random.normal(0, 0.1))

        return PredictionResponse(
            timestamp=data.timestamp,
            predicted_flow=predicted_flow,
            predicted_pressure=data.pressure * (1 + np.random.normal(0, 0.05)),
            confidence=0.95
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/decline")
async def detect_decline(data: ProductionData):
    try:
        # Use decline detection model
        if decline_model is not None:
            is_decline = decline_model.predict(np.array([data.total_flow]).reshape(1, -1))[0]
        else:
            # Dummy decline detection
            is_decline = data.total_flow < 800

        return {
            "timestamp": data.timestamp,
            "decline_detected": bool(is_decline),
            "confidence": 0.9
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    print("Starting Oil & Gas Production AI API")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
