import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


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
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000

    # True relationship with some noise
    X = np.random.normal(1000, 200, n_samples)
    noise = np.random.normal(0, 50, n_samples)
    y_true = 0.8 * X + 100 + noise

    # Simulated predictions (slightly off to show evaluation)
    y_pred = 0.75 * X + 120 + np.random.normal(0, 60, n_samples)

    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    evaluator.print_metrics_report()

    # Create plots
    evaluator.plot_actual_vs_predicted(y_true, y_pred, "Flow Rate: Actual vs Predicted")
    evaluator.plot_residuals(y_true, y_pred)


if __name__ == "__main__":
    main()
