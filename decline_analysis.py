import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class ProductionDeclineAnalyzer:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def detect_decline_events(self, production_data):
        print("Detecting Production Decline Events")
        print("---------------------------------")

        # Calculate production changes
        production_data['flow_change'] = production_data['Total_Flow'].pct_change()

        # Prepare features for anomaly detection
        features = ['Total_Flow', 'flow_change', 'Pressure', 'Temperature']
        X = self.scaler.fit_transform(production_data[features])

        # Detect anomalies using Isolation Forest
        anomalies = self.isolation_forest.fit_predict(X)
        decline_events = (anomalies == -1) & (production_data['flow_change'] < 0)

        print(f"\nDetected {decline_events.sum()} decline events")

        return decline_events

    def cluster_decline_patterns(self, production_data, decline_events):
        print("\nClustering Decline Patterns")
        print("-------------------------")

        # Filter data for decline events
        decline_data = production_data[decline_events].copy()

        if len(decline_data) > 0:
            # Prepare features for clustering
            features = ['Total_Flow', 'Oil_Flow', 'Water_Flow', 'Gas_Flow',
                        'Pressure', 'Temperature']
            X = self.scaler.fit_transform(decline_data[features])

            # Apply DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(X)

            print(f"\nIdentified {len(np.unique(clusters))} distinct decline patterns")

            return clusters
        return None


def main():
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='H')
    n_samples = len(dates)

    np.random.seed(42)
    production_data = pd.DataFrame({
        'Time': dates,
        'Total_Flow': np.random.normal(1000, 100, n_samples) *
                      (1 - np.linspace(0, 0.3, n_samples)),  # Simulated decline
        'Oil_Flow': np.random.normal(700, 50, n_samples),
        'Water_Flow': np.random.normal(200, 30, n_samples),
        'Gas_Flow': np.random.normal(500, 80, n_samples),
        'Pressure': np.random.normal(200, 20, n_samples),
        'Temperature': np.random.normal(60, 5, n_samples)
    })

    # Initialize and run decline analysis
    analyzer = ProductionDeclineAnalyzer()
    decline_events = analyzer.detect_decline_events(production_data)
    clusters = analyzer.cluster_decline_patterns(production_data, decline_events)

    if clusters is not None:
        print("\nDecline Pattern Analysis Results:")
        print(f"Total decline events: {decline_events.sum()}")
        print(f"Number of distinct patterns: {len(np.unique(clusters))}")


if __name__ == "__main__":
    main()
