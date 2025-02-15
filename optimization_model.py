import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import time
import joblib
import os

class ProductionOptimizer:
    def __init__(self, constraints):
        self.constraints = constraints
        
    def objective_function(self, x, production_history):
        """
        Objective function to maximize production while respecting constraints
        x: Array of production parameters to optimize
        """
        total_flow, pressure, gor = x
        
        # Penalty for violating constraints
        penalty = 0
        if pressure < self.constraints['min_pressure']:
            penalty += 1000 * (self.constraints['min_pressure'] - pressure)
        if pressure > self.constraints['max_pressure']:
            penalty += 1000 * (pressure - self.constraints['max_pressure'])
        if gor < self.constraints['min_gor']:
            penalty += 1000 * (self.constraints['min_gor'] - gor)
        if gor > self.constraints['max_gor']:
            penalty += 1000 * (gor - self.constraints['max_gor'])
            
        # Calculate production value (negative because we want to maximize)
        value = -(total_flow - penalty)
        return value
    
    def optimize_production(self, production_history):
        print("Optimizing Production Parameters")
        print("-------------------------------")
        
        # Define bounds for optimization variables
        bounds = [
            (self.constraints['min_flow'], self.constraints['max_flow']),  # Flow bounds
            (self.constraints['min_pressure'], self.constraints['max_pressure']),  # Pressure bounds
            (self.constraints['min_gor'], self.constraints['max_gor'])  # GOR bounds
        ]
        
        # Run optimization using differential evolution
        start_time = time.time()
        result = differential_evolution(
            self.objective_function,
            bounds,
            args=(production_history,),
            maxiter=100,
            popsize=20,
            seed=42
        )
        end_time = time.time()
        
        if result.success:
            optimal_flow, optimal_pressure, optimal_gor = result.x
            print("\nOptimization Results:")
            print(f"Optimal Flow Rate: {optimal_flow:.2f} BLPD")
            print(f"Optimal Pressure: {optimal_pressure:.2f} bar")
            print(f"Optimal GOR: {optimal_gor:.2f} scf/bbl")
            print(f"Objective Value: {-result.fun:.2f}")
            print(f"Optimization Time: {end_time - start_time:.2f} seconds")
        else:
            print("\nOptimization failed to converge")
            
        return result, end_time - start_time

def main():
    # Define system constraints
    constraints = {
        'min_flow': 500,    # Minimum flow rate (BLPD)
        'max_flow': 2000,   # Maximum flow rate (BLPD)
        'min_pressure': 50, # Minimum pressure (bar)
        'max_pressure': 300, # Maximum pressure (bar)
        'min_gor': 100,     # Minimum GOR (scf/bbl)
        'max_gor': 2000     # Maximum GOR (scf/bbl)
    }
    
    # Load preprocessed production history
    production_history = pd.read_csv('preprocessed_data.csv')
    
    # Initialize and run optimization
    optimizer = ProductionOptimizer(constraints)
    result, optimization_time = optimizer.optimize_production(production_history)
    
    # Calculate MAE between optimized and actual production
    actual_production = production_history['Total_Flow'].mean()
    optimized_production = result.x[0]
    mae = np.abs(actual_production - optimized_production)
    
    # Save results
    results = {
        'MAE': mae,
        'Optimization Time': optimization_time,
        'Optimal Flow': result.x[0],
        'Optimal Pressure': result.x[1],
        'Optimal GOR': result.x[2],
        'Objective Value': -result.fun
    }
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_path = os.path.join(results_dir, 'optimization_results.joblib')
    joblib.dump(results, results_path)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
