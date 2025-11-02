import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from nn_model import PowerConsumptionPredictor

def load_model_and_scaler():
    """Load the trained model and scaler"""
    # Load scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PowerConsumptionPredictor(input_size=5)
    model.load_state_dict(torch.load('nn_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    return model, scaler

def make_predictions(model, scaler, data):
    """Make predictions using the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocess data
    data_scaled = scaler.transform(data)
    
    # Convert to tensor and move to device
    data_tensor = torch.FloatTensor(data_scaled).to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(data_tensor)
    
    # Convert back to CPU numpy array
    predictions = predictions.cpu().numpy().flatten()
    
    return predictions

def predict_from_csv(csv_file_path, num_samples=1000):
    """Make predictions on data from CSV file"""
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(csv_file_path)
    
    # Parse datetime
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    
    # Extract time-based features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    
    # Select features (same as used in training)
    feature_columns = ['hour', 'day_of_week', 'month', 'ac_meter_0_v', 'ac_meter_0_i']
    
    # Use only a subset for demonstration (to avoid memory issues)
    sample_df = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    # Prepare features
    X = sample_df[feature_columns].values
    
    # Handle missing values
    X = X[~np.isnan(X).any(axis=1)]
    
    # Make predictions
    print("Making predictions...")
    predictions = make_predictions(model, scaler, X)
    
    # Add predictions to dataframe
    sample_df = sample_df.iloc[:len(predictions)]  # Adjust to match predictions length
    sample_df['predicted_total_kw'] = predictions
    
    # Display some results
    print("\nSample Predictions:")
    print(sample_df[['time', 'total_kw', 'predicted_total_kw']].head(10))
    
    return sample_df

if __name__ == "__main__":
    # Make predictions on the original dataset
    results_df = predict_from_csv('../influx_data.csv', num_samples=1000)
    
    # Save results
    results_df.to_csv('predictions_results.csv', index=False)
    print("\nPredictions saved to 'predictions_results.csv'")