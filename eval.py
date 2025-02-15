import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import load_model
import joblib
import time

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def calculate_metrics(self, y_true, y_pred):
        """Calculate various performance metrics"""
        self.metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }
        
        return self.metrics
    
    def plot_actual_vs_predicted(self, y_true, y_pred, title="Actual vs Predicted"):
        """Create actual vs predicted plot"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
    def plot_residuals(self, y_true, y_pred):
        """Create residual plot"""
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.tight_layout()
        plt.show()
        
    def print_metrics_report(self):
        """Print formatted metrics report"""
        print("\nModel Performance Metrics")
        print("-" * 30)
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")

def main():
    # Load the model and scaler
    model = load_model('models/flow_prediction_model')
    scaler = joblib.load('models/scaler.joblib')
    
    # Load test data (assuming you've split and saved test data separately)
    test_data = pd.read_csv('test_data.csv')
    
    # Prepare test data
    X_test = test_data.drop(['Time', 'Total_Flow'], axis=1)
    y_test = test_data['Total_Flow']
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test_scaled.reshape(X_test_scaled.shape[0], 1, -1))
    end_time = time.time()
    prediction_time = end_time - start_time
    
    # Inverse transform predictions
    y_pred = scaler.inverse_transform(y_pred)[:, 0]  # Assuming Total_Flow is the first column
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_test, y_pred)
    evaluator.print_metrics_report()
    
    # Create plots
    evaluator.plot_actual_vs_predicted(y_test, y_pred, "Flow Rate: Actual vs Predicted")
    evaluator.plot_residuals(y_test, y_pred)
    
    # Print additional information
    print(f"\nPrediction Time: {prediction_time:.4f} seconds")
    print(f"Average Prediction Time per Sample: {prediction_time/len(y_test):.6f} seconds")
    
    # Save results
    results = {
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'R2': metrics['R2'],
        'Prediction Time': prediction_time,
        'Avg Prediction Time per Sample': prediction_time/len(y_test)
    }
    joblib.dump(results, 'results/model_evaluation_results.joblib')
    print("\nResults saved to results/model_evaluation_results.joblib")

if __name__ == "__main__":
    main()
