import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_preprocessing_gpu import load_and_preprocess_data
from nn_model import PowerConsumptionPredictor, train_model

def main():
    """Main function to train the GPU-accelerated model"""
    print("Starting GPU-accelerated model training...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load and preprocess data
    train_loader, test_loader, scaler, feature_names = load_and_preprocess_data()
    
    # Train the model
    print("Starting model training...")
    model, metrics = train_model(train_loader, test_loader, num_epochs=30, learning_rate=0.001)
    
    # Print final metrics
    print("\nFinal Model Metrics:")
    print("=" * 40)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    
    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\nTraining completed successfully!")
    print("Model saved as 'nn_model.pth'")
    print("Scaler saved as 'scaler.pkl'")
    print("Training history plot saved as 'training_history.png'")
    print("Predictions plot saved as 'nn_predictions.png'")

if __name__ == "__main__":
    main()