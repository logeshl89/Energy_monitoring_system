import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pickle
import warnings
warnings.filterwarnings('ignore')

# Define the neural network model
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

def load_and_clean_data(sample_size=100000):
    """
    Load and clean the energy consumption data
    """
    print("Loading dataset...")
    # Load a sample of the data to manage memory
    df = pd.read_csv('../influx_data.csv', nrows=sample_size)
    print(f"Loaded {len(df)} rows of data")
    
    # Parse datetime
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    
    # Extract time-based features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    
    # Select relevant features for energy consumption prediction
    feature_columns = [
        'hour', 'day_of_week', 'month',
        'Average_Voltage_LN', 'Average_Current', 'Average_PF',
        'Current_I1', 'Current_I2', 'Current_I3',
        'Voltage_V1N', 'Voltage_V2N', 'Voltage_V3N',
        'PF1', 'PF2', 'PF3'
    ]
    
    target_column = 'Total_KW'
    
    # Check which features exist in the dataset
    existing_features = [col for col in feature_columns if col in df.columns]
    print(f"Using {len(existing_features)} features: {existing_features}")
    
    # Prepare data
    print("Cleaning data...")
    # Select features and target
    X = df[existing_features]
    y = df[target_column]
    
    # Handle missing values
    X = X.fillna(0)
    y = y.fillna(0)
    
    # Remove infinite values
    X = X.replace([np.inf, -np.inf], 0)
    y = y.replace([np.inf, -np.inf], 0)
    
    # Remove rows where target is still invalid
    valid_mask = np.isfinite(y) & (y != 0)
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Also ensure all features are finite
    feature_mask = np.isfinite(X).all(axis=1)
    X = X[feature_mask]
    y = y[feature_mask]
    
    print(f"Data shape after cleaning: X={X.shape}, y={y.shape}")
    
    if len(X) == 0:
        raise ValueError("No valid data remaining after cleaning")
    
    return X.values, y.values, existing_features

def prepare_data_for_training(X, y, test_size=0.2):
    """
    Prepare data for training with scaling and train/test split
    """
    print("Preparing data for training...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    return train_loader, test_loader, scaler

def train_model(train_loader, test_loader, input_size, num_epochs=30):
    """
    Train the GPU-accelerated model
    """
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Initialize model
    model = EnergyConsumptionPredictor(input_size).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.6f}')
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.6f}')
        
        # Evaluate on test set every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_loss = evaluate_model(model, test_loader, criterion, device)
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.6f}')
    
    return model

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on test data
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches

def main():
    """
    Main function to process data, train model, and save results
    """
    print("Starting complete GPU-accelerated model training...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    try:
        # Load and clean data
        X, y, feature_names = load_and_clean_data(sample_size=50000)
        
        # Prepare data for training
        train_loader, test_loader, scaler = prepare_data_for_training(X, y)
        
        # Train the model
        print("Starting model training...")
        model = train_model(train_loader, test_loader, len(feature_names), num_epochs=20)
        
        # Save the model
        torch.save(model.state_dict(), 'gpu_energy_model.pth')
        print("Model saved as 'gpu_energy_model.pth'")
        
        # Save the scaler
        with open('gpu_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("Scaler saved as 'gpu_scaler.pkl'")
        
        # Save feature names
        with open('gpu_feature_names.txt', 'w') as f:
            for feature in feature_names:
                f.write(f"{feature}\n")
        print("Feature names saved as 'gpu_feature_names.txt'")
        
        print("\nTraining completed successfully!")
        print(f"Model features: {len(feature_names)}")
        print("Files created:")
        print("  - gpu_energy_model.pth (trained model)")
        print("  - gpu_scaler.pkl (feature scaler)")
        print("  - gpu_feature_names.txt (feature names)")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()