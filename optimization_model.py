import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd


class ProductionOptimizer:
    def __init__(self, constraints):
        self.constraints = constraints

    def objective_function(self, x, production_history):
        """
        Objective function to maximize production while respecting constraints
        x: Array of production parameters to optimize
        """
        total_flow = x[0]
        pressure = x[1]

        # Penalty for violating constraints
        penalty = 0
        if pressure < self.constraints['min_pressure']:
            penalty += 1000 * (self.constraints['min_pressure'] - pressure)
        if pressure > self.constraints['max_pressure']:
            penalty += 1000 * (pressure - self.constraints['max_pressure'])

        # Calculate production value (negative because we want to maximize)
        value = -(total_flow - penalty)
        return value

    def optimize_production(self, production_history):
        print("Optimizing Production Parameters")
        print("-------------------------------")

        # Define bounds for optimization variables
        bounds = [
            (self.constraints['min_flow'], self.constraints['max_flow']),  # Flow bounds
            (self.constraints['min_pressure'], self.constraints['max_pressure'])  # Pressure bounds
        ]

        # Run optimization using differential evolution
        result = differential_evolution(
            self.objective_function,
            bounds,
            args=(production_history,),
            maxiter=100,
            popsize=20,
            seed=42
        )

        if result.success:
            optimal_flow, optimal_pressure = result.x
            print("\nOptimization Results:")
            print(f"Optimal Flow Rate: {optimal_flow:.2f} BLPD")
            print(f"Optimal Pressure: {optimal_pressure:.2f} bar")
            print(f"Objective Value: {-result.fun:.2f}")
        else:
            print("\nOptimization failed to converge")

        return result


def main():
    # Define system constraints
    constraints = {
        'min_flow': 500,  # Minimum flow rate (BLPD)
        'max_flow': 2000,  # Maximum flow rate (BLPD)
        'min_pressure': 50,  # Minimum pressure (bar)
        'max_pressure': 300  # Maximum pressure (bar)
    }

    # Create sample production history
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='H')
    n_samples = len(dates)

    production_history = pd.DataFrame({
        'Time': dates,
        'Total_Flow': np.random.normal(1000, 100, n_samples),
        'Pressure': np.random.normal(200, 20, n_samples)
    })

    # Initialize and run optimization
    optimizer = ProductionOptimizer(constraints)
    result = optimizer.optimize_production(production_history)


if __name__ == "__main__":
    main()
