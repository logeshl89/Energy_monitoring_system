import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import json
import base64
import warnings
warnings.filterwarnings('ignore')

# Define the neural network model (same as training)
class EnergyConsumptionPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        super(EnergyConsumptionPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def load_model_and_scaler():
    """
    Load the trained model and scaler
    """
    print("Loading trained GPU model and scaler...")
    
    # Load scaler
    with open('data_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model with same architecture as training
    model = EnergyConsumptionPredictor(input_size=15)  # 15 features
    model.load_state_dict(torch.load('gpu_energy_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    return model, scaler, device

def create_sample_data(voltage, current, power_factor, num_samples=1000):
    """
    Create sample data with the provided parameters
    """
    print(f"Generating {num_samples} samples with Voltage: {voltage}V, Current: {current}A, Power Factor: {power_factor}")
    
    # Generate time-based features (sample different times)
    hours = np.random.randint(0, 24, num_samples)
    days_of_week = np.random.randint(0, 7, num_samples)
    months = np.random.randint(1, 13, num_samples)
    
    # Create feature arrays with the provided values
    avg_voltage_ln = np.full(num_samples, voltage)
    avg_current = np.full(num_samples, current)
    avg_pf = np.full(num_samples, power_factor)
    
    # For other features, we'll use typical values or sample from the dataset distribution
    current_i1 = np.full(num_samples, current * 0.95)  # Assume slightly less than average
    current_i2 = np.full(num_samples, current * 1.02)  # Assume slightly more than average
    current_i3 = np.full(num_samples, current * 0.98)  # Assume close to average
    
    voltage_v1n = np.full(num_samples, voltage * 0.98)
    voltage_v2n = np.full(num_samples, voltage * 1.01)
    voltage_v3n = np.full(num_samples, voltage * 0.99)
    
    pf1 = np.full(num_samples, power_factor * 0.95)
    pf2 = np.full(num_samples, power_factor * 1.02)
    pf3 = np.full(num_samples, power_factor * 0.98)
    
    # Create feature matrix (15 features as used in training)
    features = np.column_stack([
        hours, days_of_week, months,
        avg_voltage_ln, avg_current, avg_pf,
        current_i1, current_i2, current_i3,
        voltage_v1n, voltage_v2n, voltage_v3n,
        pf1, pf2, pf3
    ])
    
    return features, hours, days_of_week

def make_predictions(model, scaler, device, features):
    """
    Make predictions using the trained model
    """
    print("Making predictions with GPU model...")
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Convert to tensor and move to device
    features_tensor = torch.FloatTensor(features_scaled).to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(features_tensor)
    
    # Convert predictions back to CPU numpy array
    y_pred = predictions.cpu().numpy().flatten()
    
    return y_pred

def generate_text_based_visualization(voltage, current, power_factor, predictions, hours, days_of_week):
    """
    Generate text-based visualization similar to what might be in comprehensive_random_testing_results.png
    """
    print("\n" + "="*80)
    print("GPU MODEL COMPREHENSIVE VISUALIZATION")
    print("="*80)
    
    # Input parameters
    print(f"\nINPUT PARAMETERS:")
    print(f"  Voltage: {voltage} V")
    print(f"  Current: {current} A")
    print(f"  Power Factor: {power_factor}")
    
    # Prediction statistics
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    min_pred = np.min(predictions)
    max_pred = np.max(predictions)
    
    print(f"\nPREDICTION STATISTICS:")
    print(f"  Mean Energy Consumption: {mean_pred:.4f} kW")
    print(f"  Standard Deviation: {std_pred:.4f} kW")
    print(f"  Minimum: {min_pred:.4f} kW")
    print(f"  Maximum: {max_pred:.4f} kW")
    print(f"  Total Samples: {len(predictions)}")
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'hour': hours,
        'day_of_week': days_of_week,
        'prediction': predictions
    })
    
    # Hourly analysis
    print(f"\nHOURLY ENERGY CONSUMPTION ANALYSIS:")
    print("  Hour |  Count |   Mean  |   Std   |   Min   |   Max  ")
    print("  -----|--------|---------|---------|---------|---------")
    
    hourly_stats = df.groupby('hour')['prediction'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    for _, row in hourly_stats.iterrows():
        hour = int(row['hour'])
        count = int(row['count'])
        mean_val = row['mean']
        std_val = row['std'] if not np.isnan(row['std']) else 0
        min_val = row['min']
        max_val = row['max']
        print(f"   {hour:2d}  | {count:6d} | {mean_val:7.4f} | {std_val:7.4f} | {min_val:7.4f} | {max_val:7.4f}")
    
    # Day of week analysis
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"\nDAY-OF-WEEK ENERGY CONSUMPTION ANALYSIS:")
    print("  Day |  Count |   Mean  |   Std   |   Min   |   Max  ")
    print("  ----|--------|---------|---------|---------|---------")
    
    day_stats = df.groupby('day_of_week')['prediction'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    for _, row in day_stats.iterrows():
        day_idx = int(row['day_of_week'])
        day_name = day_names[day_idx]
        count = int(row['count'])
        mean_val = row['mean']
        std_val = row['std'] if not np.isnan(row['std']) else 0
        min_val = row['min']
        max_val = row['max']
        print(f"  {day_name:3s} | {count:6d} | {mean_val:7.4f} | {std_val:7.4f} | {min_val:7.4f} | {max_val:7.4f}")
    
    # Create ASCII charts
    print(f"\nASCII CHARTS:")
    
    # Hourly average chart
    hourly_avg = df.groupby('hour')['prediction'].mean()
    max_val = hourly_avg.max()
    min_val = hourly_avg.min()
    range_val = max_val - min_val if max_val != min_val else 1
    
    print(f"\nHourly Average Energy Consumption:")
    for hour in range(24):
        if hour in hourly_avg.index:
            value = hourly_avg[hour]
            bar_length = int(((value - min_val) / range_val) * 40)
            bar = '█' * bar_length
            print(f"  {hour:2d}: {bar} ({value:.4f} kW)")
        else:
            print(f"  {hour:2d}: (no data)")
    
    # Distribution information
    print(f"\nENERGY CONSUMPTION DISTRIBUTION:")
    print(f"  Percentile | Value (kW)")
    print(f"  ----------|----------")
    for p in [5, 25, 50, 75, 95]:
        value = np.percentile(predictions, p)
        print(f"     {p:2d}%     |  {value:.4f}")
    
    print("="*80)

def save_visualization_data(voltage, current, power_factor, predictions, hours, days_of_week):
    """
    Save visualization data to files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed predictions
    df = pd.DataFrame({
        'timestamp': timestamp,
        'hour': hours,
        'day_of_week': days_of_week,
        'voltage_input': voltage,
        'current_input': current,
        'power_factor_input': power_factor,
        'predicted_energy_kW': predictions
    })
    
    csv_filename = f'gpu_model_visualization_data_{timestamp}.csv'
    df.to_csv(csv_filename, index=False)
    
    # Save summary statistics
    summary_data = {
        'timestamp': [timestamp],
        'input_voltage': [voltage],
        'input_current': [current],
        'input_power_factor': [power_factor],
        'mean_prediction_kw': [np.mean(predictions)],
        'std_deviation_kw': [np.std(predictions)],
        'min_prediction_kw': [np.min(predictions)],
        'max_prediction_kw': [np.max(predictions)],
        'total_samples': [len(predictions)]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f'gpu_model_visualization_summary_{timestamp}.csv'
    summary_df.to_csv(summary_filename, index=False)
    
    # Save detailed statistics as JSON
    stats = {
        'input_parameters': {
            'voltage': voltage,
            'current': current,
            'power_factor': power_factor
        },
        'prediction_statistics': {
            'mean_kw': float(np.mean(predictions)),
            'std_dev_kw': float(np.std(predictions)),
            'min_kw': float(np.min(predictions)),
            'max_kw': float(np.max(predictions)),
            'total_samples': len(predictions)
        }
    }
    
    json_filename = f'gpu_model_visualization_stats_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return csv_filename, summary_filename, json_filename

def create_simple_chart_data(predictions, hours):
    """
    Create simple chart data that could be used for visualization
    """
    # Group by hour and calculate statistics
    df = pd.DataFrame({'hour': hours, 'prediction': predictions})
    hourly_stats = df.groupby('hour')['prediction'].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    # Create time series-like data
    time_series_data = {
        'timestamps': [f"Hour {h}" for h in range(24)],
        'hourly_averages': [0.0] * 24,
        'hourly_mins': [0.0] * 24,
        'hourly_maxs': [0.0] * 24
    }
    
    for _, row in hourly_stats.iterrows():
        hour = int(row['hour'])
        time_series_data['hourly_averages'][hour] = float(row['mean'])
        time_series_data['hourly_mins'][hour] = float(row['min'])
        time_series_data['hourly_maxs'][hour] = float(row['max'])
    
    return time_series_data

def predict_and_visualize(voltage, current, power_factor, num_samples=1000):
    """
    Main function to predict and create comprehensive visualization
    """
    try:
        # Load model and scaler
        model, scaler, device = load_model_and_scaler()
        
        # Create sample data
        features, hours, days_of_week = create_sample_data(voltage, current, power_factor, num_samples)
        
        # Make predictions
        predictions = make_predictions(model, scaler, device, features)
        
        # Generate text-based visualization
        generate_text_based_visualization(voltage, current, power_factor, predictions, hours, days_of_week)
        
        # Save visualization data
        csv_file, summary_file, json_file = save_visualization_data(
            voltage, current, power_factor, predictions, hours, days_of_week)
        
        # Create chart data
        chart_data = create_simple_chart_data(predictions, hours)
        
        print(f"\nVISUALIZATION FILES GENERATED:")
        print(f"  Detailed Data: {csv_file}")
        print(f"  Summary: {summary_file}")
        print(f"  Statistics: {json_file}")
        
        # Print chart data summary
        print(f"\nCHART DATA SUMMARY:")
        avg_values = [v for v in chart_data['hourly_averages'] if v > 0]
        if avg_values:
            print(f"  Peak Hour: {chart_data['hourly_averages'].index(max(avg_values)):2d} ({max(avg_values):.4f} kW)")
            print(f"  Lowest Hour: {chart_data['hourly_averages'].index(min(avg_values)):2d} ({min(avg_values):.4f} kW)")
        
        return predictions, chart_data, csv_file, summary_file, json_file
        
    except Exception as e:
        print(f"Error during prediction and visualization: {e}")
        raise

def main():
    """
    Main function with example usage
    """
    print("GPU Model Comprehensive Visualization")
    print("="*40)
    
    # Example values (typical for energy consumption)
    voltage = 230.0    # Volts
    current = 10.0     # Amps
    power_factor = 0.9 # Power factor
    
    print(f"Using sample values: Voltage={voltage}V, Current={current}A, Power Factor={power_factor}")
    
    # Run prediction and visualization
    predictions, chart_data, csv_file, summary_file, json_file = predict_and_visualize(
        voltage, current, power_factor, num_samples=1000)
    
    print(f"\nComprehensive visualization completed successfully!")

if __name__ == "__main__":
    main()