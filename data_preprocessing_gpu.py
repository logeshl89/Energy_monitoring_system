import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess data for GPU training"""
    print("Loading dataset...")
    # Load the dataset
    df = pd.read_csv('../influx_data.csv')
    
    # Parse datetime
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    
    # Extract time-based features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    
    # Select features for modeling (using a subset for demonstration)
    # In a real scenario, you might want to use more features
    feature_columns = ['hour', 'day_of_week', 'month', 'ac_meter_0_v', 'ac_meter_0_i']
    
    # Target variable (total power consumption)
    target_column = 'total_kw'
    
    # Handle missing values
    df = df.dropna(subset=feature_columns + [target_column])
    
    # Prepare features and target
    X = df[feature_columns].values
    y = df[target_column].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors and move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # Save scaler for later use
    import pickle
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open('feature_names.txt', 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")
    
    print(f"Data preprocessing completed:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {len(feature_columns)}")
    
    return train_loader, test_loader, scaler, feature_columns

if __name__ == "__main__":
    train_loader, test_loader, scaler, feature_names = load_and_preprocess_data()
    print("Data preprocessing completed successfully!")