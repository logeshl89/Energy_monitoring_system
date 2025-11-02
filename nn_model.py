import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class PowerConsumptionPredictor(nn.Module):
    """Neural Network model for power consumption prediction"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
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

def train_model(train_loader, test_loader, num_epochs=50, learning_rate=0.001):
    """Train the neural network model on GPU"""
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Initialize model
    model = PowerConsumptionPredictor(input_size=5).to(device)  # 5 features
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    test_losses = []
    
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
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.6f}')
        
        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        
        # Evaluate on test set
        test_loss = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_loss:.6f}, Test Loss: {test_loss:.6f}')
    
    # Final evaluation
    final_metrics = evaluate_model_detailed(model, train_loader, test_loader, device)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return model, final_metrics

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model on data loader"""
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

def evaluate_model_detailed(model, train_loader, test_loader, device):
    """Detailed evaluation of the model"""
    model.eval()
    
    # Predictions on training set
    train_predictions = []
    train_targets = []
    
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_predictions.extend(output.cpu().numpy())
            train_targets.extend(target.cpu().numpy())
    
    # Predictions on test set
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_predictions.extend(output.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    # Convert to numpy arrays
    train_predictions = np.array(train_predictions).flatten()
    train_targets = np.array(train_targets).flatten()
    test_predictions = np.array(test_predictions).flatten()
    test_targets = np.array(test_targets).flatten()
    
    # Calculate metrics
    train_mse = mean_squared_error(train_targets, train_predictions)
    train_mae = mean_absolute_error(train_targets, train_predictions)
    train_r2 = r2_score(train_targets, train_predictions)
    
    test_mse = mean_squared_error(test_targets, test_predictions)
    test_mae = mean_absolute_error(test_targets, test_predictions)
    test_r2 = r2_score(test_targets, test_predictions)
    
    print("\nModel Performance:")
    print("Training Set:")
    print(f"  Mean Squared Error: {train_mse:.6f}")
    print(f"  Mean Absolute Error: {train_mae:.6f}")
    print(f"  R² Score: {train_r2:.6f}")
    
    print("Test Set:")
    print(f"  Mean Squared Error: {test_mse:.6f}")
    print(f"  Mean Absolute Error: {test_mae:.6f}")
    print(f"  R² Score: {test_r2:.6f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(test_targets, test_predictions, alpha=0.5)
    plt.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], 'r--', lw=2)
    plt.xlabel('Actual Total KW')
    plt.ylabel('Predicted Total KW')
    plt.title('Neural Network - Actual vs Predicted')
    plt.tight_layout()
    plt.savefig('nn_predictions.png')
    plt.show()
    
    # Save model
    torch.save(model.state_dict(), 'nn_model.pth')
    print("Model saved as 'nn_model.pth'")
    
    return {
        'train_mse': train_mse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }

if __name__ == "__main__":
    # This script is meant to be imported and used by the main training script
    print("Neural network model definition ready!")