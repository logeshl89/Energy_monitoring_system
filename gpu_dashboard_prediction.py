import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
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
    print(f"Creating sample data with Voltage: {voltage}, Current: {current}, Power Factor: {power_factor}")
    
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

def create_comprehensive_visualization(voltage, current, power_factor, predictions, hours, days_of_week):
    """
    Create comprehensive visualization similar to comprehensive_random_testing_results.png
    """
    print("Creating comprehensive visualization...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Energy Consumption Prediction Dashboard\nInput: Voltage={voltage}V, Current={current}A, Power Factor={power_factor}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Time Series Plot of Predictions
    axes[0, 0].plot(range(len(predictions)), predictions, 'b-', linewidth=1.5)
    axes[0, 0].set_title('Predicted Energy Consumption Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Predicted kW')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Hourly Distribution
    df = pd.DataFrame({
        'hour': hours,
        'prediction': predictions
    })
    hourly_avg = df.groupby('hour')['prediction'].mean()
    
    axes[0, 1].bar(hourly_avg.index, hourly_avg.values, color='green', alpha=0.7)
    axes[0, 1].set_title('Average Energy Consumption by Hour', fontweight='bold')
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Average Predicted kW')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribution of Predictions
    axes[1, 0].hist(predictions, bins=30, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribution of Energy Consumption Predictions', fontweight='bold')
    axes[1, 0].set_xlabel('Predicted kW')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Statistical Summary
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    min_pred = np.min(predictions)
    max_pred = np.max(predictions)
    
    # Create a text summary
    stats_text = f"""Statistical Summary:
    
Mean Prediction: {mean_pred:.4f} kW
Standard Deviation: {std_pred:.4f} kW
Minimum Prediction: {min_pred:.4f} kW
Maximum Prediction: {max_pred:.4f} kW
Total Samples: {len(predictions)}

Input Parameters:
Voltage: {voltage} V
Current: {current} A
Power Factor: {power_factor}
"""
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    axes[1, 1].set_title('Prediction Statistics', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    output_filename = f'gpu_model_dashboard_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as '{output_filename}'")
    return output_filename

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
        
        # Create visualization
        output_file = create_comprehensive_visualization(voltage, current, power_factor, 
                                                       predictions, hours, days_of_week)
        
        # Print summary statistics
        print("\n" + "="*50)
        print("PREDICTION SUMMARY")
        print("="*50)
        print(f"Input Voltage: {voltage} V")
        print(f"Input Current: {current} A")
        print(f"Input Power Factor: {power_factor}")
        print(f"Mean Predicted Energy Consumption: {np.mean(predictions):.4f} kW")
        print(f"Min Predicted Energy Consumption: {np.min(predictions):.4f} kW")
        print(f"Max Predicted Energy Consumption: {np.max(predictions):.4f} kW")
        print(f"Standard Deviation: {np.std(predictions):.4f} kW")
        print("="*50)
        
        return predictions, output_file
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

def main():
    """
    Main function - example usage
    """
    print("GPU Model Energy Consumption Prediction Dashboard")
    print("="*50)
    
    # Example values (you can change these)
    voltage = 230.0  # Volts
    current = 10.0   # Amps
    power_factor = 0.9  # Power factor
    
    print(f"Using sample values: Voltage={voltage}V, Current={current}A, Power Factor={power_factor}")
    
    # Make predictions and create visualization
    predictions, output_file = predict_energy_consumption(voltage, current, power_factor)
    
    print(f"\nDashboard visualization created: {output_file}")

if __name__ == "__main__":
    main()