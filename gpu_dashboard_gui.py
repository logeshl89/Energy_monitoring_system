import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import json
import threading

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

class GPUEnergyDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("GPU Energy Consumption Prediction Dashboard")
        self.root.geometry("800x500")
        self.root.resizable(True, True)
        
        # Variables
        self.voltage_var = tk.DoubleVar(value=230.0)
        self.current_var = tk.DoubleVar(value=10.0)
        self.power_factor_var = tk.DoubleVar(value=0.9)
        self.samples_var = tk.IntVar(value=100)
        self.model = None
        self.scaler = None
        self.device = None
        
        # Create UI
        self.create_widgets()
        
        # Load model
        self.load_model_async()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="GPU Energy Consumption Prediction", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input parameters frame
        input_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        input_frame.columnconfigure(1, weight=1)
        
        # Voltage
        ttk.Label(input_frame, text="Voltage (V):").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        voltage_entry = ttk.Entry(input_frame, textvariable=self.voltage_var, width=20)
        voltage_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Current
        ttk.Label(input_frame, text="Current (A):").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        current_entry = ttk.Entry(input_frame, textvariable=self.current_var, width=20)
        current_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Power Factor
        ttk.Label(input_frame, text="Power Factor:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10))
        pf_entry = ttk.Entry(input_frame, textvariable=self.power_factor_var, width=20)
        pf_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Number of samples
        ttk.Label(input_frame, text="Sample Size:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10))
        samples_entry = ttk.Entry(input_frame, textvariable=self.samples_var, width=20)
        samples_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=(0, 20))
        
        # Predict button
        self.predict_button = ttk.Button(button_frame, text="Predict Energy Consumption", 
                                        command=self.predict_energy_consumption)
        self.predict_button.grid(row=0, column=0, padx=(0, 10))
        
        # Load model button
        self.load_button = ttk.Button(button_frame, text="Reload Model", 
                                     command=self.load_model_async)
        self.load_button.grid(row=0, column=1, padx=(0, 10))
        
        # Save results button
        self.save_button = ttk.Button(button_frame, text="Save Results", 
                                     command=self.save_results, state=tk.DISABLED)
        self.save_button.grid(row=0, column=2)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        status_frame.columnconfigure(0, weight=1)
        
        # Status text
        self.status_var = tk.StringVar(value="Loading model...")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     foreground="blue")
        self.status_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Results text
        self.results_text = tk.Text(results_frame, height=10, wrap=tk.WORD)
        results_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure weights for resizing
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
    
    def load_model_async(self):
        """Load model in a separate thread to prevent UI freezing"""
        self.status_var.set("Loading model...")
        self.predict_button.config(state=tk.DISABLED)
        threading.Thread(target=self.load_model, daemon=True).start()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            # Load scaler
            with open('data_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize model with same architecture as training
            self.model = EnergyConsumptionPredictor(input_size=15)  # 15 features
            self.model.load_state_dict(torch.load('gpu_energy_model.pth', map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # Update UI
            self.root.after(0, self.on_model_loaded)
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            self.root.after(0, lambda: self.on_model_error(error_msg))
    
    def on_model_loaded(self):
        """Called when model is successfully loaded"""
        self.status_var.set(f"Model loaded successfully! Using device: {self.device}")
        self.predict_button.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, f"GPU Model loaded successfully!\nDevice: {self.device}\n\n")
    
    def on_model_error(self, error_msg):
        """Called when there's an error loading the model"""
        self.status_var.set("Error loading model")
        self.predict_button.config(state=tk.DISABLED)
        self.results_text.insert(tk.END, f"Error: {error_msg}\n")
        messagebox.showerror("Model Load Error", error_msg)
    
    def create_sample_data(self, voltage, current, power_factor, num_samples):
        """Create sample data with the provided parameters"""
        # Generate time-based features (sample different times)
        hours = np.random.randint(0, 24, num_samples)
        days_of_week = np.random.randint(0, 7, num_samples)
        months = np.random.randint(1, 13, num_samples)
        
        # Create feature arrays with the provided values
        avg_voltage_ln = np.full(num_samples, voltage)
        avg_current = np.full(num_samples, current)
        avg_pf = np.full(num_samples, power_factor)
        
        # For other features, we'll use typical values
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
    
    def make_predictions(self, features):
        """Make predictions using the trained model"""
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Convert to tensor and move to device
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(features_tensor)
        
        # Convert predictions back to CPU numpy array
        y_pred = predictions.cpu().numpy().flatten()
        
        return y_pred
    
    def predict_energy_consumption(self):
        """Main prediction function"""
        try:
            # Get input values
            voltage = self.voltage_var.get()
            current = self.current_var.get()
            power_factor = self.power_factor_var.get()
            num_samples = self.samples_var.get()
            
            # Validate inputs
            if voltage <= 0 or current <= 0 or power_factor <= 0 or power_factor > 1:
                messagebox.showerror("Invalid Input", "Please enter valid positive values. Power factor should be between 0 and 1.")
                return
            
            if num_samples <= 0 or num_samples > 10000:
                messagebox.showerror("Invalid Input", "Sample size should be between 1 and 10000.")
                return
            
            # Update status
            self.status_var.set("Making predictions...")
            self.predict_button.config(state=tk.DISABLED)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Predicting energy consumption...\n")
            self.results_text.insert(tk.END, f"Input: Voltage={voltage}V, Current={current}A, Power Factor={power_factor}\n")
            self.results_text.insert(tk.END, f"Sample Size: {num_samples}\n\n")
            self.root.update()
            
            # Create sample data
            features, hours, days_of_week = self.create_sample_data(voltage, current, power_factor, num_samples)
            
            # Make predictions
            predictions = self.make_predictions(features)
            
            # Generate results
            self.generate_results(voltage, current, power_factor, predictions, hours, days_of_week)
            
            # Enable save button
            self.save_button.config(state=tk.NORMAL)
            self.predictions_data = {
                'voltage': voltage,
                'current': current,
                'power_factor': power_factor,
                'predictions': predictions,
                'hours': hours,
                'days_of_week': days_of_week
            }
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            self.results_text.insert(tk.END, f"Error: {error_msg}\n")
            messagebox.showerror("Prediction Error", error_msg)
        finally:
            self.status_var.set("Ready")
            self.predict_button.config(state=tk.NORMAL)
    
    def generate_results(self, voltage, current, power_factor, predictions, hours, days_of_week):
        """Generate and display results"""
        # Calculate statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        min_pred = np.min(predictions)
        max_pred = np.max(predictions)
        
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'hour': hours,
            'day_of_week': days_of_week,
            'prediction': predictions
        })
        
        # Hourly analysis
        hourly_avg = df.groupby('hour')['prediction'].mean().reset_index()
        hourly_avg = hourly_avg.sort_values('prediction', ascending=False)
        
        # Day of week analysis
        day_avg = df.groupby('day_of_week')['prediction'].mean().reset_index()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_avg['day_name'] = day_avg['day_of_week'].map(lambda x: day_names[x])
        day_avg = day_avg.sort_values('prediction', ascending=False)
        
        # Display results
        self.results_text.insert(tk.END, "PREDICTION RESULTS\n")
        self.results_text.insert(tk.END, "="*50 + "\n")
        self.results_text.insert(tk.END, f"Input Parameters:\n")
        self.results_text.insert(tk.END, f"  Voltage: {voltage} V\n")
        self.results_text.insert(tk.END, f"  Current: {current} A\n")
        self.results_text.insert(tk.END, f"  Power Factor: {power_factor}\n\n")
        
        self.results_text.insert(tk.END, f"Prediction Statistics:\n")
        self.results_text.insert(tk.END, f"  Mean Energy Consumption: {mean_pred:.4f} kW\n")
        self.results_text.insert(tk.END, f"  Standard Deviation: {std_pred:.4f} kW\n")
        self.results_text.insert(tk.END, f"  Minimum: {min_pred:.4f} kW\n")
        self.results_text.insert(tk.END, f"  Maximum: {max_pred:.4f} kW\n")
        self.results_text.insert(tk.END, f"  Total Samples: {len(predictions)}\n\n")
        
        self.results_text.insert(tk.END, f"Top 5 Hours by Energy Consumption:\n")
        for i, (_, row) in enumerate(hourly_avg.head().iterrows()):
            self.results_text.insert(tk.END, f"  {i+1}. Hour {int(row['hour']):2d}: {row['prediction']:.4f} kW\n")
        
        self.results_text.insert(tk.END, f"\nEnergy Consumption by Day of Week:\n")
        for _, row in day_avg.iterrows():
            self.results_text.insert(tk.END, f"  {row['day_name']:10s}: {row['prediction']:.4f} kW\n")
        
        self.results_text.insert(tk.END, "\nPrediction completed successfully!\n")
    
    def save_results(self):
        """Save results to files"""
        try:
            if not hasattr(self, 'predictions_data'):
                messagebox.showwarning("No Data", "No prediction data to save.")
                return
            
            # Get data
            data = self.predictions_data
            voltage = data['voltage']
            current = data['current']
            power_factor = data['power_factor']
            predictions = data['predictions']
            hours = data['hours']
            days_of_week = data['days_of_week']
            
            # Ask for save location
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"gpu_dashboard_results_{timestamp}"
            )
            
            if not filename:
                return
            
            # Save detailed results
            df = pd.DataFrame({
                'timestamp': timestamp,
                'hour': hours,
                'day_of_week': days_of_week,
                'voltage_input': voltage,
                'current_input': current,
                'power_factor_input': power_factor,
                'predicted_energy_kW': predictions
            })
            
            df.to_csv(filename, index=False)
            
            # Save summary
            summary_data = {
                'timestamp': [timestamp],
                'input_voltage': [voltage],
                'input_current': [current],
                'input_power_factor': [power_factor],
                'mean_prediction_kw': [np.mean(predictions)],
                'std_deviation_kw': [np.std(predictions)],
                'min_prediction_kw': [np.min(predictions)],
                'max_prediction_kw': [np.max(predictions)],
                'total_samples': [len(predictions)]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_filename = filename.replace('.csv', '_summary.csv')
            summary_df.to_csv(summary_filename, index=False)
            
            self.results_text.insert(tk.END, f"\nResults saved to:\n  {filename}\n  {summary_filename}\n")
            messagebox.showinfo("Save Successful", f"Results saved successfully!\n\nFiles:\n{filename}\n{summary_filename}")
            
        except Exception as e:
            error_msg = f"Error saving results: {str(e)}"
            self.results_text.insert(tk.END, f"Error: {error_msg}\n")
            messagebox.showerror("Save Error", error_msg)

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = GPUEnergyDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()