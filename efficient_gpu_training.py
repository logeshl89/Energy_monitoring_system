import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

class PowerConsumptionPredictor(nn.Module):
    """Neural Network model for power consumption prediction"""
    
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout_rate=0.2):
        super(PowerConsumptionPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
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

def load_and_preprocess_data_sample(sample_size=100000):
    """Load and preprocess a sample of the data for GPU training"""
    print("Loading dataset...")
    # Load the dataset
    df = pd.read_csv('../influx_data.csv')
    
    # Take a sample to reduce memory usage
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Parse datetime
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    
    # Extract time-based features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    
    # Select features for modeling
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
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Save scaler for later use
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

def train_model(train_loader, test_loader, num_epochs=20, learning_rate=0.001):
    """Train the neural network model on GPU"""
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Initialize model
    model = PowerConsumptionPredictor(input_size=5).to(device)  # 5 features
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
    
    # Save model
    torch.save(model.state_dict(), 'nn_model.pth')
    print("Model saved as 'nn_model.pth'")
    
    return model

def main():
    """Main function to train the GPU-accelerated model"""
    print("Starting GPU-accelerated model training...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load and preprocess data (using a sample)
    train_loader, test_loader, scaler, feature_names = load_and_preprocess_data_sample(sample_size=50000)
    
    # Train the model
    print("Starting model training...")
    model = train_model(train_loader, test_loader, num_epochs=15, learning_rate=0.001)
    
    print("\nTraining completed successfully!")
    print("Model saved as 'nn_model.pth'")
    print("Scaler saved as 'scaler.pkl'")

if __name__ == "__main__":
    main()