import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os

def create_simple_visualization_data(voltage, current, power_factor, predictions, hours):
    """
    Create simple visualization data without matplotlib
    """
    print("Creating simple visualization data...")
    
    # Group predictions by hour for hourly analysis
    df = pd.DataFrame({
        'hour': hours,
        'prediction': predictions
    })
    
    # Calculate hourly averages
    hourly_avg = df.groupby('hour')['prediction'].mean().reset_index()
    
    # Create a simple text-based visualization
    print("\n" + "="*60)
    print("ENERGY CONSUMPTION PREDICTION RESULTS")
    print("="*60)
    print(f"Input Parameters:")
    print(f"  Voltage: {voltage} V")
    print(f"  Current: {current} A")
    print(f"  Power Factor: {power_factor}")
    print(f"\nPrediction Statistics:")
    print(f"  Mean: {np.mean(predictions):.4f} kW")
    print(f"  Std Dev: {np.std(predictions):.4f} kW")
    print(f"  Min: {np.min(predictions):.4f} kW")
    print(f"  Max: {np.max(predictions):.4f} kW")
    print(f"  Total Samples: {len(predictions)}")
    
    print(f"\nHourly Average Energy Consumption:")
    print("Hour | Avg kW")
    print("-----|-------")
    for _, row in hourly_avg.iterrows():
        print(f"  {int(row['hour']):2d} | {row['prediction']:.4f}")
    
    # Create a simple ASCII chart
    print(f"\nSimple Visualization (ASCII Chart):")
    max_pred = hourly_avg['prediction'].max()
    min_pred = hourly_avg['prediction'].min()
    range_pred = max_pred - min_pred if max_pred != min_pred else 1
    
    for _, row in hourly_avg.iterrows():
        hour = int(row['hour'])
        value = row['prediction']
        # Create a simple bar using asterisks
        bar_length = int(((value - min_pred) / range_pred) * 20)
        bar = '*' * bar_length
        print(f"  {hour:2d}: {bar} ({value:.4f} kW)")
    
    print("="*60)

def load_and_visualize_latest_predictions():
    """
    Load the latest prediction results and create visualization
    """
    # Find the latest prediction files
    files = [f for f in os.listdir('.') if f.startswith('gpu_model_predictions_') and f.endswith('.csv')]
    
    if not files:
        print("No prediction files found!")
        return
    
    # Get the latest file
    latest_file = sorted(files)[-1]
    print(f"Loading predictions from: {latest_file}")
    
    # Load the data
    df = pd.read_csv(latest_file)
    
    # Extract parameters and predictions
    voltage = df['voltage_input'].iloc[0]
    current = df['current_input'].iloc[0]
    power_factor = df['power_factor_input'].iloc[0]
    predictions = df['predicted_energy_kW'].values
    hours = df['hour'].values
    
    # Create visualization
    create_simple_visualization_data(voltage, current, power_factor, predictions, hours)

def main():
    """
    Main function to create simple visualization
    """
    print("GPU Model Simple Visualization")
    print("="*30)
    
    # Load and visualize latest predictions
    load_and_visualize_latest_predictions()

if __name__ == "__main__":
    main()