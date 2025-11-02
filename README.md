# GPU-Accelerated Energy Consumption Prediction

This project implements a neural network model for predicting energy consumption using GPU acceleration with PyTorch. It includes data preprocessing, model training, evaluation, and a web-based dashboard for real-time predictions.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Web Dashboard](#web-dashboard)
- [Output Files](#output-files)
- [License](#license)

## Features

- GPU-accelerated neural network training using PyTorch
- Data preprocessing with scikit-learn
- Real-time energy consumption prediction dashboard
- Automatic model evaluation and visualization
- Support for various training approaches (minimal, efficient, complete)
- Interactive GUI for model monitoring

## Requirements

- Python 3.7+
- PyTorch with CUDA support
- NVIDIA GPU with CUDA support

## Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── data_preprocessing_gpu.py      # Data loading and preprocessing for GPU training
├── nn_model.py                    # Neural network model definition and training functions
├── gpu_training.py                # Main training script
├── predict_gpu.py                 # Script for making predictions with the trained model
├── web_dashboard.py               # Flask web server for real-time predictions dashboard
├── gpu_dashboard_gui.py           # PyQt-based GUI dashboard
├── interactive_gpu_dashboard.py   # Interactive dashboard with enhanced features
├── gpu_energy_dashboard.py        # Energy-specific dashboard implementation
├── gpu_dashboard_prediction.py    # Prediction-focused dashboard component
├── gpu_model_visualization.py     # Model visualization tools
├── simple_visualization.py        # Simplified visualization utilities
├── gpu_prediction_csv.py          # CSV export functionality for predictions
├── data_cleaning.py               # Data cleaning utilities
├── quick_gpu_training.py          # Fast training implementation
├── efficient_gpu_training.py      # Efficient training approach
├── complete_gpu_training.py       # Full-featured training implementation
├── minimal_training.py            # Minimal training setup
├── simple_gpu_training.py         # Simple training approach
├── fixed_training.py              # Fixed training parameters version
├── templates/
│   └── dashboard.html             # Web dashboard HTML template
├── requirements.txt               # Required Python packages
└── README.md                     # This file
```

## Usage

### Basic Training

To train the model with default settings:

```bash
python gpu_training.py
```

### Different Training Approaches

The project offers several training scripts for different scenarios:

- `minimal_training.py`: Minimal setup for quick experimentation
- `quick_gpu_training.py`: Fast training with reduced epochs
- `simple_gpu_training.py`: Simplified training approach
- `efficient_gpu_training.py`: Optimized for efficiency
- `complete_gpu_training.py`: Full-featured training with extensive evaluation

Example:

```bash
python quick_gpu_training.py
```

### Making Predictions

After training, use the model to make predictions:

```bash
python predict_gpu.py
```

### Running the Web Dashboard

Start the real-time prediction dashboard:

```bash
python web_dashboard.py
```

Then access the dashboard at `http://localhost:5000`

### Running the GUI Dashboard

Launch the PyQt-based GUI dashboard:

```bash
python gpu_dashboard_gui.py
```

## Model Architecture

The neural network consists of:
- Input layer with 5-15 features (depending on implementation)
- 3 hidden layers (128, 64, 32 neurons)
- ReLU activation functions
- Dropout for regularization (rate=0.2)
- Single output neuron for energy/power consumption prediction

The model is specifically designed for predicting energy consumption based on:
- Time-based features (hour, day of week, month)
- Electrical parameters (voltage, current, power factor)
- Multi-phase measurements (current and voltage for each phase)

## Web Dashboard

The web dashboard (`web_dashboard.py`) provides:
- Real-time energy consumption predictions
- Automatic simulation with data generation
- Visualization of prediction distributions
- Hourly and daily consumption patterns
- Model validation results
- Responsive web interface with interactive charts

Features:
- Live updating every 5 seconds
- Comprehensive statistical analysis
- Multiple chart types (bar, line, radar)
- Health check endpoint at `/health`
- Simulation status endpoint at `/simulation/status`

## Output Files

Training produces several output files:

- `gpu_energy_model.pth`: Trained model weights
- `data_scaler.pkl`: Feature scaler for preprocessing
- `training_history.png`: Training/validation loss curves
- `nn_predictions.png`: Actual vs predicted values plot
- `predictions_results.csv`: Sample predictions on dataset
- Various log and temporary files

## License

This project is licensed for educational and research purposes.