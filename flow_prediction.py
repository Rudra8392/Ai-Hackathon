import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# Sample code to demonstrate the implementation
def load_and_preprocess_data():
    print("1. Data Loading and Preprocessing")
    print("--------------------------------")

    # Simulated data loading (replace with actual file paths)
    print("Loading datasets...")
    try:
        # In practice, you would use:
         lp_data = pd.read_excel("Platform MC_Low Pressure System Monitoring.xlsb")
         well_data = pd.read_excel("Data_Platform MC - Copy.xlsm")
         test_data = pd.read_excel("Well_Tests.xlsx")

        # For demonstration, creating sample dataframe
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='H')
        n_samples = len(dates)

        sample_data = pd.DataFrame({
            'Time': dates,
            'Total_Flow': np.random.normal(1000, 100, n_samples),
            'Oil_Flow': np.random.normal(700, 50, n_samples),
            'Water_Flow': np.random.normal(200, 30, n_samples),
            'Gas_Flow': np.random.normal(500, 80, n_samples),
            'Pressure': np.random.normal(200, 20, n_samples),
            'Temperature': np.random.normal(60, 5, n_samples)
        })

        print("\nSample data shape:", sample_data.shape)
        print("\nFirst few rows of sample data:")
        print(sample_data.head())

        return sample_data

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
        Dense(6)  # Predicting 6 parameters (flows, pressure, temperature)
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

        # Train the model (using small number of epochs for demonstration)
        print("\nTraining the model...")
        history = model.fit(
            X_train, y_train,
            epochs=2,  # Increased for actual training
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        print("\nTraining completed. The model can now be used for predictions.")


if __name__ == "__main__":
    main()
