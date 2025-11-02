import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
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

def save_predictions_to_csv(voltage, current, power_factor, predictions, hours, days_of_week):
    """
    Save predictions to CSV file
    """
    print("Saving predictions to CSV file...")
    
    # Create DataFrame with predictions and parameters
    df = pd.DataFrame({
        'hour': hours,
        'day_of_week': days_of_week,
        'voltage_input': voltage,
        'current_input': current,
        'power_factor_input': power_factor,
        'predicted_energy_kW': predictions
    })
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'gpu_model_predictions_{timestamp}.csv'
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    # Calculate and display statistics
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    min_pred = np.min(predictions)
    max_pred = np.max(predictions)
    
    print(f"\nPrediction Results:")
    print(f"  Mean: {mean_pred:.4f} kW")
    print(f"  Std Dev: {std_pred:.4f} kW")
    print(f"  Min: {min_pred:.4f} kW")
    print(f"  Max: {max_pred:.4f} kW")
    print(f"  Total Samples: {len(predictions)}")
    
    print(f"\nResults saved to: {filename}")
    
    # Also save summary statistics
    summary_data = {
        'timestamp': [timestamp],
        'input_voltage': [voltage],
        'input_current': [current],
        'input_power_factor': [power_factor],
        'mean_prediction_kw': [mean_pred],
        'std_deviation_kw': [std_pred],
        'min_prediction_kw': [min_pred],
        'max_prediction_kw': [max_pred],
        'total_samples': [len(predictions)]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f'gpu_model_summary_{timestamp}.csv'
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary saved to: {summary_filename}")
    
    return filename, summary_filename

def predict_energy_consumption(voltage, current, power_factor):
    """
    Main function to predict energy consumption based on input parameters
    """
    try:
        # Load model and scaler
        model, scaler, device = load_model_and_scaler()
        
        # Create sample data
        features, hours, days_of_week = create_sample_data(voltage, current, power_factor)
        
        # Make predictions
        predictions = make_predictions(model, scaler, device, features)
        
        # Save to CSV
        csv_file, summary_file = save_predictions_to_csv(voltage, current, power_factor, 
                                                        predictions, hours, days_of_week)
        
        return predictions, csv_file, summary_file
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

def main():
    """
    Main function with example usage
    """
    print("GPU Model Energy Consumption Prediction")
    print("="*40)
    
    # Example values (you can change these)
    voltage = 230.0  # Volts
    current = 10.0   # Amps
    power_factor = 0.9  # Power factor
    
    print(f"Using sample values: Voltage={voltage}V, Current={current}A, Power Factor={power_factor}")
    
    # Make predictions and save to CSV
    predictions, csv_file, summary_file = predict_energy_consumption(voltage, current, power_factor)
    
    print(f"\nPrediction completed successfully!")
    print(f"Detailed results: {csv_file}")
    print(f"Summary: {summary_file}")

if __name__ == "__main__":
    main()