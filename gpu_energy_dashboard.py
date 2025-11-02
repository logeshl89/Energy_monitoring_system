import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import json
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

def create_sample_data(voltage, current, power_factor, num_samples=100):
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

def generate_comprehensive_report(voltage, current, power_factor, predictions, hours, days_of_week):
    """
    Generate a comprehensive report similar to comprehensive_random_testing_results.png
    """
    print("Generating comprehensive report...")
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'hour': hours,
        'day_of_week': days_of_week,
        'prediction': predictions
    })
    
    # Calculate statistics
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
        },
        'hourly_analysis': {},
        'day_of_week_analysis': {}
    }
    
    # Hourly analysis
    hourly_avg = df.groupby('hour')['prediction'].agg(['mean', 'std', 'min', 'max']).reset_index()
    for _, row in hourly_avg.iterrows():
        hour = int(row['hour'])
        stats['hourly_analysis'][hour] = {
            'mean': float(row['mean']),
            'std': float(row['std']),
            'min': float(row['min']),
            'max': float(row['max'])
        }
    
    # Day of week analysis
    day_avg = df.groupby('day_of_week')['prediction'].agg(['mean', 'std', 'min', 'max']).reset_index()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for _, row in day_avg.iterrows():
        day_idx = int(row['day_of_week'])
        day_name = day_names[day_idx]
        stats['day_of_week_analysis'][day_name] = {
            'mean': float(row['mean']),
            'std': float(row['std']),
            'min': float(row['min']),
            'max': float(row['max'])
        }
    
    return stats

def save_report_and_data(voltage, current, power_factor, predictions, hours, days_of_week, stats):
    """
    Save comprehensive report and data
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
    
    csv_filename = f'gpu_dashboard_detailed_results_{timestamp}.csv'
    df.to_csv(csv_filename, index=False)
    
    # Save statistics as JSON
    json_filename = f'gpu_dashboard_statistics_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save summary as CSV
    summary_data = {
        'timestamp': [timestamp],
        'input_voltage': [voltage],
        'input_current': [current],
        'input_power_factor': [power_factor],
        'mean_prediction_kw': [stats['prediction_statistics']['mean_kw']],
        'std_deviation_kw': [stats['prediction_statistics']['std_dev_kw']],
        'min_prediction_kw': [stats['prediction_statistics']['min_kw']],
        'max_prediction_kw': [stats['prediction_statistics']['max_kw']],
        'total_samples': [stats['prediction_statistics']['total_samples']]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f'gpu_dashboard_summary_{timestamp}.csv'
    summary_df.to_csv(summary_filename, index=False)
    
    return csv_filename, json_filename, summary_filename

def display_results(stats):
    """
    Display results in a formatted way
    """
    print("\n" + "="*70)
    print("GPU MODEL ENERGY CONSUMPTION PREDICTION DASHBOARD")
    print("="*70)
    
    # Input parameters
    params = stats['input_parameters']
    print(f"\nINPUT PARAMETERS:")
    print(f"  Voltage: {params['voltage']} V")
    print(f"  Current: {params['current']} A")
    print(f"  Power Factor: {params['power_factor']}")
    
    # Prediction statistics
    pred_stats = stats['prediction_statistics']
    print(f"\nPREDICTION STATISTICS:")
    print(f"  Mean Energy Consumption: {pred_stats['mean_kw']:.4f} kW")
    print(f"  Standard Deviation: {pred_stats['std_dev_kw']:.4f} kW")
    print(f"  Minimum: {pred_stats['min_kw']:.4f} kW")
    print(f"  Maximum: {pred_stats['max_kw']:.4f} kW")
    print(f"  Total Samples: {pred_stats['total_samples']}")
    
    # Hourly analysis (show top 5 hours)
    print(f"\nTOP 5 HOURS BY AVERAGE ENERGY CONSUMPTION:")
    hourly_data = [(hour, data['mean']) for hour, data in stats['hourly_analysis'].items()]
    hourly_data.sort(key=lambda x: x[1], reverse=True)
    for i, (hour, mean_kw) in enumerate(hourly_data[:5]):
        print(f"  {i+1}. Hour {hour:2d}: {mean_kw:.4f} kW")
    
    # Day of week analysis (show all days)
    print(f"\nENERGY CONSUMPTION BY DAY OF WEEK:")
    day_data = [(day, data['mean']) for day, data in stats['day_of_week_analysis'].items()]
    day_data.sort(key=lambda x: x[1], reverse=True)
    for day, mean_kw in day_data:
        print(f"  {day:10s}: {mean_kw:.4f} kW")
    
    print("="*70)

def predict_energy_consumption_dashboard(voltage, current, power_factor):
    """
    Main function to create comprehensive dashboard predictions
    """
    try:
        # Load model and scaler
        model, scaler, device = load_model_and_scaler()
        
        # Create sample data
        features, hours, days_of_week = create_sample_data(voltage, current, power_factor)
        
        # Make predictions
        predictions = make_predictions(model, scaler, device, features)
        
        # Generate comprehensive report
        stats = generate_comprehensive_report(voltage, current, power_factor, 
                                            predictions, hours, days_of_week)
        
        # Save report and data
        csv_file, json_file, summary_file = save_report_and_data(
            voltage, current, power_factor, predictions, hours, days_of_week, stats)
        
        # Display results
        display_results(stats)
        
        print(f"\nFILES GENERATED:")
        print(f"  Detailed Results: {csv_file}")
        print(f"  Statistics Report: {json_file}")
        print(f"  Summary: {summary_file}")
        
        return stats, csv_file, json_file, summary_file
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

def main():
    """
    Main function - example with sample values
    """
    print("GPU Model Energy Consumption Dashboard")
    print("="*40)
    
    # Example values (typical for energy consumption)
    voltage = 230.0    # Volts
    current = 10.0     # Amps
    power_factor = 0.9 # Power factor
    
    print(f"Using sample values: Voltage={voltage}V, Current={current}A, Power Factor={power_factor}")
    
    # Run the dashboard prediction
    stats, csv_file, json_file, summary_file = predict_energy_consumption_dashboard(
        voltage, current, power_factor)
    
    print(f"\nDashboard prediction completed successfully!")

if __name__ == "__main__":
    main()