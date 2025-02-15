import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import Sequential
from tensorflow import LSTM, Dense, Dropout
import joblib
import os
import time

def load_and_preprocess_data():
    print("1. Data Loading and Preprocessing")
    print("--------------------------------")
    
    try:
        # Load actual data
        lp_data = pd.read_excel("Platform MC_Low Pressure System Monitoring.xlsb")
        well_data = pd.read_excel("Data_Platform MC - Copy.xlsm")
        test_data = pd.read_excel("Well_Tests.xlsx")
        
        # Merge datasets based on common columns (you may need to adjust this based on your actual data structure)
        merged_data = pd.merge(lp_data, well_data, on='Time', how='outer')
        merged_data = pd.merge(merged_data, test_data, on='Date', how='outer')
        
        # Handle missing values
        merged_data = merged_data.interpolate()
        
        # Convert time to datetime if not already
        merged_data['Time'] = pd.to_datetime(merged_data['Time'])
        
        # Sort by time
        merged_data = merged_data.sort_values('Time')
        
        # Select relevant features (adjust based on your data)
        features = ['Total_Flow', 'Oil_Flow', 'Water_Flow', 'Gas_Flow', 'Pressure', 'Temperature', 'GOR']
        data = merged_data[['Time'] + features]
        
        print("\nMerged data shape:", data.shape)
        print("\nFirst few rows of merged data:")
        print(data.head())
        
        return data
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def prepare_sequences(data, seq_length=24):
    """Prepare sequences for LSTM model"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:(i + seq_length)]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
        
    return np.array(sequences), np.array(targets)

def create_lstm_model(input_shape):
    """Create LSTM model for flow prediction"""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(7)  # Predicting 7 parameters (flows, pressure, temperature, GOR)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    # Load and preprocess data
    data = load_and_preprocess_data()
    
    if data is not None:
        print("\n2. Feature Engineering and Model Preparation")
        print("------------------------------------------")
        
        # Normalize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data.drop('Time', axis=1))
        
        # Prepare sequences for LSTM
        seq_length = 24  # Use 24 hours of data to predict next hour
        X, y = prepare_sequences(scaled_data, seq_length)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        print(f"\nTraining set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Create and compile the model
        print("\n3. Model Creation and Training")
        print("-----------------------------")
        model = create_lstm_model((seq_length, X.shape[2]))
        print("\nModel Summary:")
        model.summary()
        
        # Train the model
        print("\nTraining the model...")
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\nTraining completed. Training time: {training_time:.2f} seconds")
        
        # Evaluate the model
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest MAE: {test_mae:.4f}")
        
        # Save the model
        model_dir = 'models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'flow_prediction_model')
        model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save the scaler
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
        # Save evaluation metrics
        metrics = {
            'MAE': test_mae,
            'Training Time': training_time,
            'Test Loss': test_loss
        }
        metrics_path = os.path.join(model_dir, 'metrics.joblib')
        joblib.dump(metrics, metrics_path)
        print(f"Evaluation metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
