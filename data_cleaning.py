import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

def clean_energy_data(input_file='../influx_data.csv', sample_size=50000):
    """
    Clean and preprocess the energy consumption dataset
    """
    print("Loading dataset...")
    # Load a sample of the data to manage memory
    df = pd.read_csv(input_file, nrows=sample_size)
    print(f"Loaded {len(df)} rows of data")
    
    # Parse datetime
    print("Parsing datetime...")
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
    
    # Use kW1 as target since Total_KW has all NaN values
    target_column = 'kW1'
    print(f"Using {target_column} as target variable")
    
    # Check which features exist in the dataset
    existing_features = [col for col in feature_columns if col in df.columns]
    print(f"Available features: {existing_features}")
    
    # Data cleaning process
    print("Cleaning data...")
    
    # 1. Handle missing values in features
    for col in existing_features:
        if col in df.columns and df[col].isnull().sum() > 0:
            # Fill with median for numerical columns
            df[col] = df[col].fillna(df[col].median())
            print(f"  Filled missing values in {col} with median")
    
    # 2. Handle missing values in target
    initial_target_missing = df[target_column].isnull().sum()
    if initial_target_missing > 0:
        df[target_column] = df[target_column].fillna(df[target_column].median())
        print(f"  Filled {initial_target_missing} missing values in {target_column} with median")
    
    # 3. Remove infinite values
    initial_rows = len(df)
    columns_to_check = existing_features + [target_column]
    # Create a mask for finite values across all columns
    finite_mask = pd.Series([True] * len(df))
    for col in columns_to_check:
        if col in df.columns:
            finite_mask = finite_mask & np.isfinite(df[col])
    df = df[finite_mask]
    rows_after_inf_removal = len(df)
    print(f"  Removed {initial_rows - rows_after_inf_removal} rows with infinite values")
    
    # 4. Remove rows with all zero features (if any)
    if len(df) > 0:
        feature_data = df[existing_features]
        # Check for all zero rows
        rows_all_zero = (feature_data == 0).all(axis=1).sum()
        if rows_all_zero > 0:
            df = df[~(feature_data == 0).all(axis=1)]
            print(f"  Removed {rows_all_zero} rows with all zero features")
    
    # 5. Remove outliers using IQR method for target variable
    if len(df) > 0:
        target_series = df[target_column]
        # Convert to pandas Series if it's not already
        if not isinstance(target_series, pd.Series):
            target_series = pd.Series(target_series)
        Q1 = target_series.quantile(0.25)
        Q3 = target_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        initial_rows = len(df)
        df = df[(df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)]
        rows_after_outlier_removal = len(df)
        print(f"  Removed {initial_rows - rows_after_outlier_removal} outlier rows")
    
    # 6. Final check for valid data
    if len(df) > 0:
        # Ensure we only use columns that exist
        if isinstance(df, pd.DataFrame):
            existing_features = [col for col in existing_features if col in df.columns]
        X = df[existing_features]
        y = df[target_column]
        
        # Ensure all data is finite
        valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid_mask]
        y = y[valid_mask]
    else:
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
    
    print(f"Final data shape: X={X.shape}, y={y.shape}")
    
    if len(X) == 0:
        raise ValueError("No valid data remaining after cleaning")
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save cleaned data
    print("Saving cleaned data...")
    np.save('X_train_cleaned.npy', np.array(X_train_scaled))
    np.save('X_test_cleaned.npy', np.array(X_test_scaled))
    # Convert to numpy arrays for saving
    y_train_array = np.array(y_train.values if isinstance(y_train, pd.Series) else y_train)
    y_test_array = np.array(y_test.values if isinstance(y_test, pd.Series) else y_test)
    np.save('y_train_cleaned.npy', y_train_array)
    np.save('y_test_cleaned.npy', y_test_array)
    
    # Save scaler
    with open('data_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open('cleaned_feature_names.txt', 'w') as f:
        for feature in existing_features:
            f.write(f"{feature}\n")
    
    print("Data cleaning completed successfully!")
    print(f"  Training set: {np.array(X_train_scaled).shape}")
    print(f"  Test set: {np.array(X_test_scaled).shape}")
    print(f"  Features used: {len(existing_features)}")
    print("  Files saved:")
    print("    - X_train_cleaned.npy")
    print("    - X_test_cleaned.npy")
    print("    - y_train_cleaned.npy")
    print("    - y_test_cleaned.npy")
    print("    - data_scaler.pkl")
    print("    - cleaned_feature_names.txt")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, existing_features

def analyze_data_quality(input_file='../influx_data.csv', sample_size=10000):
    """
    Analyze the quality of the dataset
    """
    print("Analyzing data quality...")
    # Load a sample of the data
    df = pd.read_csv(input_file, nrows=sample_size)
    
    # Parse datetime
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    
    # Target column analysis (using kW1 since Total_KW is all NaN)
    target_column = 'kW1'
    print(f"\n{target_column} analysis:")
    print(f"  Total rows: {len(df)}")
    print(f"  Missing values: {df[target_column].isnull().sum()}")
    print(f"  Infinite values: {np.isinf(df[target_column]).sum() if target_column in df.columns else 0}")
    print(f"  Zero values: {(df[target_column] == 0).sum() if target_column in df.columns else 0}")
    if target_column in df.columns and df[target_column].count() > 0:
        print(f"  Min value: {df[target_column].min()}")
        print(f"  Max value: {df[target_column].max()}")
        print(f"  Mean value: {df[target_column].mean():.4f}")
        print(f"  Std deviation: {df[target_column].std():.4f}")
    
    # Feature columns analysis
    feature_columns = [
        'Average_Voltage_LN', 'Average_Current', 'Average_PF',
        'Current_I1', 'Current_I2', 'Current_I3',
        'Voltage_V1N', 'Voltage_V2N', 'Voltage_V3N',
        'PF1', 'PF2', 'PF3'
    ]
    
    print(f"\nFeature analysis:")
    for col in feature_columns:
        if col in df.columns:
            missing = df[col].isnull().sum()
            infinite = np.isinf(df[col]).sum() if df[col].dtype in [np.float64, np.int64, np.float32, np.int32] else 0
            print(f"  {col}: {missing} missing, {infinite} infinite")

if __name__ == "__main__":
    # Analyze data quality first
    analyze_data_quality()
    
    # Clean the data
    clean_energy_data()