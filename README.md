# Oil & Gas Production AI System

An intelligent system for monitoring, analyzing, and optimizing oil and gas production using machine learning and real-time data analytics.

## 🌟 Live Demo

Check out the live monitoring dashboard:
[Production Monitoring Dashboard](https://ai-hackathon-kjgcgp5bttkeyooucznr8v.streamlit.app/)


![WhatsApp Image 2025-02-15 at 21 13 29_8f327002](https://github.com/user-attachments/assets/63d986a1-391c-4527-a8fb-9b4573286112)

## 🚀 Features

- Real-time production monitoring and visualization
- AI-powered flow prediction using LSTM neural networks
- Production decline analysis and pattern detection
- Automated optimization of production parameters
- RESTful API for model deployment and integration
- Interactive monitoring dashboard built with Streamlit

## 💁️ Project Structure

```
├── Data_Platform MC - Copy.xlsm    # Platform production data
├── Platform MC_Low Pressure System Monitoring.xlsb
├── Well_Tests.xlsx                 # Well test data
├── api_deployment.py               # FastAPI service deployment
├── decline_analysis.py            # Production decline analysis
├── eval.py                        # Model evaluation scripts
├── flow_prediction.py             # LSTM-based flow prediction
├── monitoring_dashboard.py        # Streamlit dashboard
├── optimization_model.py          # Production optimization
├── preprocessed_data.csv          # Processed dataset
└── requirements.txt               # Project dependencies
```

## 🛠️ Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   git clone https://github.com/Rudra8392/Ai-Hackathon
   cd Ai-Hackathon
   pip install -r requirements.txt
   ```

## 🔧 Components

### 1. Flow Prediction Model

- LSTM-based neural network for production forecasting
- Features: Total flow, oil flow, water flow, gas flow, pressure, temperature, and GOR
- Sequence-based prediction with 24-hour lookback window

### 2. Decline Analysis

- Automated detection of production decline events
- Pattern clustering using DBSCAN
- Anomaly detection with Isolation Forest
- Performance metrics calculation and visualization

### 3. Production Optimization

- Differential evolution algorithm for parameter optimization
- Constraint-based optimization of flow rates, pressure, and GOR
- Real-time optimization suggestions

### 4. Monitoring Dashboard

- Real-time visualization of production parameters
- Key Performance Indicators (KPIs)
- Interactive date range selection
- Historical data analysis

### 5. API Service

- FastAPI-based REST API
- Real-time prediction endpoints
- Production data validation
- Error handling and logging

### 6. Model Evaluation (eval.py)

- Evaluation of LSTM model performance
- Metrics calculated: MAE, RMSE, R2
- Visualizations: Actual vs Predicted plot, Residual plot
- Sample performance results:
  - MAE: 67.8183
  - RMSE: 84.7019
  - R2: 0.7280

## 📊 Model Performance

The system includes comprehensive evaluation metrics:

- Flow prediction accuracy
- Decline event detection precision
- Optimization effectiveness
- Real-time prediction latency

## 🚀 Usage

1. Start the API service:

   ```bash
   python api_deployment.py
   ```

2. Launch the monitoring dashboard:

   ```bash
   streamlit run monitoring_dashboard.py
   ```

3. Run decline analysis:

   ```bash
   python decline_analysis.py
   ```

4. Train the flow prediction model:

   ```bash
   python flow_prediction.py
   ```

5. Optimize production parameters:

   ```bash
   python optimization_model.py
   ```

6. Evaluate the model:

   ```bash
   python eval.py
   ```

## 📈 API Endpoints

- `GET /`: API health check
- `POST /predict/flow`: Production flow prediction

Example Request Body:

```json
{
  "timestamp": "2024-02-15T00:00:00",
  "total_flow": 1000.0,
  "oil_flow": 800.0,
  "water_flow": 200.0,
  "gas_flow": 50000.0,
  "pressure": 200.0,
  "temperature": 60.0,
  "gor": 62.5
}
```

## 📝 Footer

This project is designed to improve operational efficiency and decision-making in oil and gas production using AI-powered solutions. The evaluation of our flow prediction model demonstrated reliable performance with the following key metrics:

- MAE: 67.8183
- RMSE: 84.7019
- R2: 0.7280

Thank You!
