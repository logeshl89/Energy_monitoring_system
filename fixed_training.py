import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simple neural network
class PowerPredictor(nn.Module):
    def __init__(self, input_size):
        super(PowerPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def main():
    print("Loading data...")
    # Load a small sample of the data
    df = pd.read_csv('../influx_data.csv', nrows=5000)  # Only first 5000 rows
    
    # Parse datetime and extract features
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    
    # Select features (using available columns)
    features = ['hour', 'day_of_week', 'Average_Voltage_LN', 'Average_Current']
    target = 'Total_KW'
    
    # Check for infinite or NaN values
    print("Checking for NaN or infinite values...")
    for feature in features:
        nan_count = df[feature].isna().sum()
        inf_count = np.isinf(df[feature]).sum() if df[feature].dtype in [np.float64, np.int64] else 0
        print(f"{feature}: {nan_count} NaN, {inf_count} infinite")
    
    nan_count_target = df[target].isna().sum()
    inf_count_target = np.isinf(df[target]).sum() if df[target].dtype in [np.float64, np.int64] else 0
    print(f"{target}: {nan_count_target} NaN, {inf_count_target} infinite")
    
    # Remove rows with NaN or infinite values
    df = df.dropna(subset=features + [target])
    
    # Remove infinite values
    for col in features + [target]:
        if df[col].dtype in [np.float64, np.int64]:
            df = df[np.isfinite(df[col])]
    
    print(f"Data shape after cleaning: {df.shape}")
    
    # Prepare data
    X = df[features].values
    y = df[target].values
    
    # Check if we have data left
    if len(X) == 0:
        print("No valid data left after cleaning!")
        return
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X range: {X.min()} to {X.max()}")
    print(f"y range: {y.min()} to {y.max()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors and move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    
    # Check for NaN in tensors
    if torch.isnan(X_train_tensor).any() or torch.isnan(y_train_tensor).any():
        print("NaN values found in tensors!")
        return
    
    # Create model
    model = PowerPredictor(len(features)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("Training model...")
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print("NaN loss detected!")
            break
            
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/20], Loss: {loss.item():.6f}')
    
    # Save model
    torch.save(model.state_dict(), 'fixed_model.pth')
    print("Model saved as 'fixed_model.pth'")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()