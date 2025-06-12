import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load dataset function
def load_data(filepath):
    data = pd.read_csv(filepath)
    data.drop(columns=['date_hour'], inplace=True)
    # Handle missing values if necessary (e.g., fill forward/backward)
    data.fillna(method='ffill', inplace=True)
    data.interpolate(method='linear', inplace=True)
    return data.values

# Preprocess the data: Scaling, converting to numpy array
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

# Create windows for input-output pairs (for multi-step time series forecasting)
def create_sequences(data, input_steps, output_steps, stride=1):
    X, y = [], []
    for i in range(0, len(data) - input_steps - output_steps, stride):  # Note: stride
        X.append(data[i:(i + input_steps)])
        y.append(data[(i + input_steps):(i + input_steps + output_steps)])
    return np.array(X), np.array(y)
    
# Split the data into training, validation, and testing sets
def split_data(X, y, test_size=0.2, val_size=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size/(test_size + val_size))
    return X_train, X_val, X_test, y_train, y_val, y_test


# Load the dataset
filepath = './data/pems.csv'  # Update to your dataset path
data = load_data(filepath)

input_steps = 24
output_steps = 24
# Create sequences for multi-step forecasting
X, y = create_sequences(data, input_steps=input_steps, output_steps=output_steps)
    
# Split the data into training, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
scaler = MinMaxScaler(feature_range=(0, 1))
    
# Reshape to 2D for scaling
num_features = X_train.shape[2]
X_train_reshaped = X_train.reshape(-1, num_features)
X_val_reshaped = X_val.reshape(-1, num_features)
X_test_reshaped = X_test.reshape(-1, num_features)
y_train_reshaped = y_train.reshape(-1, num_features)
y_val_reshaped = y_val.reshape(-1, num_features)
y_test_reshaped = y_test.reshape(-1, num_features)
    
# Fit scaler on training data
scaler.fit(X_train_reshaped)
    
# Transform all data
X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
y_train_scaled = scaler.transform(y_train_reshaped).reshape(y_train.shape)
y_val_scaled = scaler.transform(y_val_reshaped).reshape(y_val.shape)
y_test_scaled = scaler.transform(y_test_reshaped).reshape(y_test.shape)

# augmentation_factor: Number of augmented samples to generate per original sample.
# noise_level: Standard deviation of the Gaussian noise to add.
def augment_time_series(data, augmentation_factor=2, noise_level=0.01):
    augmented_data = []    
    for sample in data:
        augmented_data.append(sample)  # Include the original sample
        
        # Generate synthetic samples
        for _ in range(augmentation_factor):
            noise = np.random.normal(loc=0.0, scale=noise_level, size=sample.shape)
            augmented_sample = sample + noise
            augmented_data.append(augmented_sample)    
    return np.array(augmented_data)

# Apply data augmentation
augmentation_factor = 2  # Generate 2 augmented samples per original sample
noise_level = 0.01       # Adjust noise level as needed

def load_data():
    X_train_scaled_a = augment_time_series(X_train_scaled, augmentation_factor, noise_level)
    y_train_scaled_a = np.repeat(y_train_scaled, augmentation_factor + 1, axis=0)
    
    # Do NOT augment validation and test sets
    X_val_scaled_a = X_val_scaled
    y_val_scaled_a = y_val_scaled
    X_test_scaled_a = X_test_scaled
    y_test_scaled_a = y_test_scaled
    
    return X_train_scaled_a, y_train_scaled_a, X_val_scaled_a, y_val_scaled_a, X_test_scaled_a, y_test_scaled_a

