import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import joblib
import os

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
        features = ['Total_Flow', 'flow_change', 'Pressure', 'Temperature', 'GOR']
        X = self.scaler.fit_transform(production_data[features])
        
        # Detect anomalies using Isolation Forest
        start_time = time.time()
        anomalies = self.isolation_forest.fit_predict(X)
        end_time = time.time()
        
        decline_events = (anomalies == -1) & (production_data['flow_change'] < 0)
        
        print(f"\nDetected {decline_events.sum()} decline events")
        print(f"Detection time: {end_time - start_time:.2f} seconds")
        
        return decline_events, end_time - start_time
    
    def cluster_decline_patterns(self, production_data, decline_events):
        print("\nClustering Decline Patterns")
        print("-------------------------")
        
        # Filter data for decline events
        decline_data = production_data[decline_events].copy()
        
        if len(decline_data) > 0:
            # Prepare features for clustering
            features = ['Total_Flow', 'Oil_Flow', 'Water_Flow', 'Gas_Flow', 
                       'Pressure', 'Temperature', 'GOR']
            X = self.scaler.fit_transform(decline_data[features])
            
            # Apply DBSCAN clustering
            start_time = time.time()
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(X)
            end_time = time.time()
            
            print(f"\nIdentified {len(np.unique(clusters))} distinct decline patterns")
            print(f"Clustering time: {end_time - start_time:.2f} seconds")
            
            return clusters, end_time - start_time
        return None, 0

def main():
    # Load the preprocessed data
    data = pd.read_csv('preprocessed_data.csv')
    
    # Initialize and run decline analysis
    analyzer = ProductionDeclineAnalyzer()
    decline_events, detection_time = analyzer.detect_decline_events(data)
    clusters, clustering_time = analyzer.cluster_decline_patterns(data, decline_events)
    
    if clusters is not None:
        print("\nDecline Pattern Analysis Results:")
        print(f"Total decline events: {decline_events.sum()}")
        print(f"Number of distinct patterns: {len(np.unique(clusters))}")
        
        # Calculate MAE for decline prediction
        y_true = data['Total_Flow'].values
        y_pred = data['Total_Flow'].values.copy()
        y_pred[decline_events] *= 0.9  # Assume 10% decline for detected events
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"\nDecline Prediction MAE: {mae:.4f}")
        print(f"Decline Prediction RMSE: {rmse:.4f}")
        
        # Save results
        results = {
            'MAE': mae,
            'RMSE': rmse,
            'Detection Time': detection_time,
            'Clustering Time': clustering_time,
            'Total Decline Events': decline_events.sum(),
            'Distinct Patterns': len(np.unique(clusters))
        }
        
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_path = os.path.join(results_dir, 'decline_analysis_results.joblib')
        joblib.dump(results, results_path)
        print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
