# Oil & Gas Production AI System

An intelligent system for monitoring, analyzing, and optimizing oil and gas production using machine learning and real-time data analytics.

## 🌟 Live Demo

Check out the live monitoring dashboard:
[Production Monitoring Dashboard](https://ai-hackathon-kjgcgp5bttkeyooucznr8v.streamlit.app/)

![Monitoring Dashboard](https://github.com/user-attachments/assets/63d986a1-391c-4527-a8fb-9b4573286112)

## 🚀 Features

- **Real-time production monitoring & visualization**
- **AI-powered flow prediction using LSTM**
- **Production decline detection & analysis**
- **Automated optimization of production parameters**
- **RESTful API for deployment & integration**
- **Interactive dashboard (Streamlit)**

## 🛠️ Project Structure

```
├── Data_Platform MC.xlsm    # Platform production data
├── Well_Tests.xlsx          # Well test data
├── api_deployment.py        # FastAPI service deployment
├── decline_analysis.py      # Production decline analysis
├── eval.py                  # Model evaluation scripts
├── flow_prediction.py       # LSTM-based flow prediction
├── monitoring_dashboard.py  # Streamlit dashboard
├── optimization_model.py    # Production optimization
├── preprocessed_data.csv    # Processed dataset
└── requirements.txt         # Project dependencies
```

## 🔧 Installation

```bash
git clone https://github.com/Rudra8392/Ai-Hackathon
cd Ai-Hackathon
pip install -r requirements.txt
```

## 📊 AI Model Components

### 1. Flow Prediction Model

- **Model:** LSTM (Long Short-Term Memory)
- **Features:** Total flow, oil flow, water flow, gas flow, pressure, temperature, and GOR
- **Training:** 4-month historical data, 2-month test window
- **Evaluation:** MAE, RMSE, R²

### 2. Decline Analysis

- **Automated decline detection** using anomaly detection (Isolation Forest, DBSCAN)
- **Root cause analysis** using classification models (Random Forest, XGBoost)
- **Pattern clustering** for similar well behaviors

### 3. Production Optimization

- **Goal:** Maximize production efficiency while minimizing bottlenecks
- **Techniques:** Genetic Algorithms, Reinforcement Learning, Constraint-Based Optimization

### 4. Monitoring Dashboard

- **Real-time visualization** of production parameters
- **Key Performance Indicators (KPIs)**
- **Historical data analysis**

### 5. API Service

- **FastAPI-based REST API** for real-time predictions
- **Endpoints:** Flow rate prediction, anomaly detection, optimization suggestions
- **Data validation & error handling**

### 6. Model Evaluation

- **Metrics:**
  - Mean Absolute Error (MAE): 67.8183
  - Root Mean Square Error (RMSE): 84.7019
  - R²: 0.7280
- **Visualizations:** Actual vs. predicted plots, residual analysis

## 🚀 Usage

1. **Start API service:**
   ```bash
   python api_deployment.py
   ```

2. **Launch monitoring dashboard:**
   ```bash
   streamlit run monitoring_dashboard.py
   ```

3. **Run decline analysis:**
   ```bash
   python decline_analysis.py
   ```

4. **Train flow prediction model:**
   ```bash
   python flow_prediction.py
   ```

5. **Optimize production parameters:**
   ```bash
   python optimization_model.py
   ```

6. **Evaluate model performance:**
   ```bash
   python eval.py
   ```

## 📡 API Endpoints

- **`GET /`** - API health check
- **`POST /predict/flow`** - Predicts production flow rates

### Example Request:
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

## 📝 Summary

This project enhances oil & gas production monitoring using AI-powered solutions. Key capabilities include real-time flow prediction, production decline detection, and automated optimization. Our LSTM model has demonstrated reliable performance, achieving:

- **MAE:** 67.8183
- **RMSE:** 84.7019
- **R²:** 0.7280

🚀 **Let's optimize oil & gas production with AI!**

