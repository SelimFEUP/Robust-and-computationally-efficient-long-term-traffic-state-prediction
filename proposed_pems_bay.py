import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
#from fancyimpute import SoftImpute
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, SpatialDropout1D, Add
from tensorflow.keras.layers import Dense, LayerNormalization
import random

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Load the PEMS-BAY dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    data.drop(columns=['date_time'], inplace=True)
    
    # Handle missing values if necessary
    data.fillna(method='ffill', inplace=True)
    data.interpolate(method='linear', inplace=True)
    return data.values

# Preprocessing
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

# Creating windows
def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data[i:(i + input_steps), :])
        y.append(data[(i + input_steps):(i + input_steps + output_steps), :])
    return np.array(X), np.array(y)

# Data spliting
def split_data(X, y, test_size=0.2, val_size=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size/(test_size + val_size))
    return X_train, X_val, X_test, y_train, y_val, y_test


# Load the dataset
filepath = 'PEMS_BAY.csv'

data = load_data(filepath)

input_steps = 24
output_steps = 24

# Creating sequences
X, y = create_sequences(data, input_steps=input_steps, output_steps=output_steps)
    
# Spliting data
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


def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    if mask is not None:
        logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    #attention_weights = tf.keras.layers.Dropout(rate=dropout)(attention_weights)
    return tf.matmul(attention_weights, value)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)
        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        return self.dense(concat_attention)

class SpatioTemporalPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, num_sensors, d_model, time_steps):
        super(SpatioTemporalPositionalEncoding, self).__init__()
        self.num_sensors = num_sensors
        self.d_model = d_model
        self.time_steps = time_steps

        # Learnable spatial embeddings for each sensor
        self.spatial_embedding = self.add_weight(
            shape=(num_sensors, d_model), initializer="random_normal", trainable=True, name="spatial_embedding"
        )

        # Temporal encoding (sinusoidal)
        self.temporal_encoding = self.compute_temporal_encoding()

    def compute_temporal_encoding(self):
        position = np.arange(self.time_steps)[:, np.newaxis]
        d_model_half = self.d_model // 2
        div_term = np.exp(np.arange(0, d_model_half) * -(np.log(10000.0) / self.d_model))

        temporal_encoding = np.zeros((self.time_steps, self.d_model))
        temporal_encoding[:, 0:d_model_half * 2:2] = np.sin(position * div_term)
        temporal_encoding[:, 1:d_model_half * 2:2] = np.cos(position * div_term)

        return tf.constant(temporal_encoding, dtype=tf.float32)

    def call(self, x):
        # Convert sparse tensor to dense if needed
        if isinstance(x, tf.SparseTensor):
           x = tf.sparse.to_dense(x)
           
        # x shape: (batch_size, time_steps, d_model)
        batch_size = tf.shape(x)[0]  # Dynamic batch size

        # Compute spatial-temporal encoding: (time_steps, num_sensors, d_model)
        spatio_temporal_encoding = (
               self.spatial_embedding[tf.newaxis, :, :]  # (1, num_sensors, d_model)
               + self.temporal_encoding[:, tf.newaxis, :]  # (time_steps, 1, d_model)
        )

        # Aggregate spatial dimension: (time_steps, num_sensors, d_model) -> (time_steps, d_model)
        spatio_temporal_encoding = tf.reduce_sum(spatio_temporal_encoding, axis=1)

        # Broadcast across batch size: (time_steps, d_model) -> (batch_size, time_steps, d_model)
        spatio_temporal_encoding = tf.broadcast_to(spatio_temporal_encoding, [batch_size, self.time_steps, self.d_model])

        # Add positional encoding to the input tensor
        return x + spatio_temporal_encoding

# Mask creation for padding
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

# Encoder Layer   
def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(time_steps, num_features), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
    
    # Pre-Normalization
    norm_inputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    attention = MultiHeadAttention(d_model, num_heads, name="attention")({
        'query': norm_inputs, 'key': norm_inputs, 'value': norm_inputs, 'mask': padding_mask
    })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention_out = inputs + attention  # Residual Connection
    
    # Feedforward Network with Gated Mechanism
    norm_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_out)
    gate = tf.keras.layers.Dense(units, activation="sigmoid")(norm_attention)
    feature = tf.keras.layers.Dense(units, activation="relu")(norm_attention)
    gated_out = gate * feature  # Gated Linear Unit
    feedforward = tf.keras.layers.Dense(units=d_model)(gated_out)
    feedforward = tf.keras.layers.Dropout(rate=dropout)(feedforward)
    outputs = attention_out + feedforward  # Residual Connection
    
    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

# Dynamic Feature Embedding
class DynamicFeatureEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, activation=None, use_dropout=False, dropout_rate=0.1, use_batch_norm=False, num_layers=1, **kwargs):
        """
        Args:
            embedding_dim (int): Dimensionality of the embedding space.
            activation (str or callable, optional): Activation function to use. Defaults to None (linear activation).
            use_dropout (bool): Whether to apply dropout. Defaults to False.
            dropout_rate (float): Dropout rate if dropout is enabled. Defaults to 0.1.
            use_batch_norm (bool): Whether to use batch normalization. Defaults to False.
            num_layers (int): Number of dense layers to use. Defaults to 1.
        """
        super(DynamicFeatureEmbedding, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.activation = tf.keras.activations.get(activation)
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers

        self.dense_layers = [
            tf.keras.layers.Dense(embedding_dim, activation=self.activation) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate) if use_dropout else None
        self.batch_norm = tf.keras.layers.BatchNormalization() if use_batch_norm else None

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
            if self.use_batch_norm:
                x = self.batch_norm(x, training=training)
            if self.use_dropout:
                x = self.dropout(x, training=training)
        return x

def hybrid_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return mse + 0.5 * mae
    
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-3, 
    first_decay_steps=1000,
    alpha=1e-4
)

def temporal_block(x, num_filters, kernel_size, dilation_rate, dropout):
    skip_connection = x  # Preserve input for the residual connection
    out = Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = SpatialDropout1D(dropout)(out)
    # Add the residual connection
    if skip_connection.shape[-1] != num_filters:
        skip_connection = Conv1D(filters=num_filters, kernel_size=1, padding='same')(skip_connection)
    out = Add()([out, skip_connection])  # Combine output and skip connection
    return Activation('relu')(out)

class DeepProjectionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_hidden, dropout_rate=0.1):
        super(DeepProjectionLayer, self).__init__()
        self.d_model = d_model
        self.num_hidden = num_hidden
        self.dropout_rate = dropout_rate

        # Components of the projection
        self.dense1 = Dense(num_hidden, activation='relu')  # Nonlinear transformation
        self.gate = Dense(num_hidden, activation='sigmoid')  # Gating mechanism matches dense1 output size
        self.scale = tf.Variable(initial_value=tf.sqrt(tf.cast(d_model, tf.float32)), trainable=True, name="learnable_scale")
        self.dense2 = Dense(d_model, activation=None)  # Linear projection back to d_model
        self.input_projection = Dense(d_model, activation=None)  # Input projection for residual
        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        # Initial nonlinear transformation
        x = self.dense1(inputs)

        # Apply gating mechanism
        gate = self.gate(inputs)
        x = x * gate  # Element-wise gating (now dimensions match)

        # Linear projection to d_model
        x = self.dense2(x)

        # Scale and normalize
        x = x * self.scale
        x = self.layer_norm(x)

        # Project inputs to match x's shape
        inputs_proj = self.input_projection(inputs)

        # Add residual connection
        x = Add()([x, inputs_proj])

        return x

# Model
def transformer(time_steps, d_model, num_heads, num_layers, units, dropout, output_size):
    input_shape = (time_steps, num_features)
    inputs = tf.keras.Input(shape=input_shape, name="inputs")
    padding_mask = tf.keras.layers.Lambda(
        lambda x: create_padding_mask(tf.reduce_sum(x, axis=-1))
    )(inputs)
    
    dynamic_features = DynamicFeatureEmbedding(input_dim=time_steps, embedding_dim=d_model)(inputs)
    x = tf.keras.layers.Concatenate(axis=-1)([inputs, dynamic_features])
    
    # Projection with learnable scaling factor
    #projection = tf.keras.layers.Dense(d_model, activation='linear')(x)
    projection = DeepProjectionLayer(d_model=d_model, num_hidden=d_model * 2)(x)
    #scale = tf.Variable(tf.sqrt(tf.cast(d_model, tf.float32)), trainable=True, name="learnable_scale")
    #projection *= scale  # Apply the learnable scaling factor
    projection *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    projection = SpatioTemporalPositionalEncoding(
        num_sensors=num_features, d_model=d_model, time_steps=time_steps
    )(projection)
    projection = tf.keras.layers.LayerNormalization()(projection)
    x = tf.keras.layers.Dropout(rate=dropout)(projection)

    for i in range(num_layers):
        x = encoder_layer(units, d_model, num_heads, dropout, name=f"encoder_layer_{i}")([x, padding_mask])
    
    x = temporal_block(x, num_filters=units, kernel_size=3, dilation_rate=1, dropout=dropout)
    x = tf.keras.layers.Dropout(rate=dropout)(x)
    x = tf.keras.layers.Dense(units=units, activation='relu', kernel_regularizer=l2(0.01))(x)
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size))(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Initializing the model
d_model = num_features
num_heads = 25
num_layers = 1
units = 256
dropout = 0.1
learning_rate = 0.0005
output_size = num_features
time_steps = 24


# Building the model
best_model = transformer(time_steps,d_model=d_model,num_heads=num_heads,num_layers=num_layers,units=units,dropout=dropout,output_size=output_size)

# Compile the model
best_model.compile(loss=hybrid_loss,optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),metrics=[tf.keras.metrics.MeanAbsoluteError()])
best_model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=10, mode='min')
mc = tf.keras.callbacks.ModelCheckpoint('proposed_pems_bay_24.weights.h5', monitor='val_mean_absolute_error', verbose=5, save_best_only=True, 
          mode='min', save_weights_only=True)

history = best_model.fit(X_train_scaled, y_train_scaled, validation_data=(X_val_scaled, y_val_scaled), epochs=100,batch_size=32,callbacks=[mc, early_stopping])

# Save training history
#df_h = pd.DataFrame.from_dict(history.history)
#df_h.to_csv('time_series_data/trial_proposed_pem_bay_24.csv')

# Load model
best_model.load_weights('proposed_pems_bay_24.weights.h5')

# Define evaluation function
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
    return mae, rmse
    
# Evaluate the model on the test set
mae, rmse = evaluate_model(best_model, X_test_scaled, y_test_scaled)
print(f"Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")

# Plotting
import matplotlib.pyplot as plt
import numpy as np

num_features = y_test_scaled.shape[-1]  

# Reshape to 2D for inverse transform
y_test_scaled_reshaped = y_test_scaled.reshape(-1, num_features)
predicted_values_scaled = best_model.predict(X_test_scaled).reshape(-1, num_features)

# Inverse transform
true_values = scaler.inverse_transform(y_test_scaled_reshaped)  # Original scale
predicted_values = scaler.inverse_transform(predicted_values_scaled)

# Reshape back to original shape if needed
true_values = true_values.reshape(y_test_scaled.shape)
predicted_values = predicted_values.reshape(y_test_scaled.shape)

# Sensors to plot (update indices based on your dataset's columns)
sensor_indices = [13, 5, 7, 11]  # Randomly selected

# Plot for each sensor
annotations = ['(a)', '(b)', '(c)', '(d)']  # Annotations for each subplot
plt.figure(figsize=(7, 7))

for i, sensor_idx in enumerate(sensor_indices):
    plt.subplot(2, 2, i + 1)  # Create a 2x2 grid for plots
    plt.scatter(
        true_values[:, 0, sensor_idx],
        predicted_values[:, 0, sensor_idx],
        color='blue',
        alpha=0.7,
        s=5  
    )
    plt.plot(
        [np.min(true_values[:, 0, sensor_idx]), np.max(true_values[:, 0, sensor_idx])],
        [np.min(true_values[:, 0, sensor_idx]), np.max(true_values[:, 0, sensor_idx])],
        color='red', linestyle='dashed'
    )  # 45° line
    #plt.title(f'Sensor {sensor_idx}')
    plt.ylabel('Predicted Values')
    
    # Set y-axis label with annotation in the middle
    plt.xlabel(f'Actual Values\n{annotations[i]}', fontsize=10)

plt.tight_layout()  # Adjust layout for better viewing
plt.show()

# Assuming sensor_indices, true_values, predicted_values, and annotations are defined
for i, sensor_idx in enumerate(sensor_indices):
    plt.subplot(2, 2, i + 1)  # Create a 2x2 grid for plots
    # Plot True values as a line
    plt.plot(true_values[:48, 0, sensor_idx], label='True', color='b', linestyle='-')  # True values in blue
    
    # Plot Predicted values as a transparent line
    plt.plot(predicted_values[:48, 0, sensor_idx], label='Predicted', color='r', linestyle='-', alpha=0.3)  # Set alpha for transparency
    
    # Plot Predicted values with markers ('*')
    plt.plot(predicted_values[:48, 0, sensor_idx], '*', color='r')  # Predicted values as red stars
    
    plt.ylabel('Traffic Speed')
    
    # Set x-axis label with annotation in the middle
    plt.xlabel(f'Time Steps\n{annotations[i]}', fontsize=10)

    # Add legend for each subplot
    plt.legend(loc='lower left')

plt.tight_layout()  # Adjust layout for better viewing
plt.show()

# Function to add noise to the dataset
def add_noise(data, noise_level):
    noise = np.random.normal(0, noise_level, data.shape)  # Gaussian noise
    noisy_data = data + noise
    return noisy_data

# Evaluate the model on the test set with various noise levels
def evaluate_with_noise(model, X_test, y_test, noise_levels):
    results = {}
    for noise_level in noise_levels:
        # Add noise to both input and output data
        noisy_X_test = add_noise(X_test, noise_level)
        noisy_y_test = add_noise(y_test, noise_level)
        
        # Evaluate the model
        mae, rmse = evaluate_model(model, noisy_X_test, noisy_y_test)
        
        # Store results
        results[noise_level] = {'MAE': mae, 'RMSE': rmse}
        print(f"Noise Level: {noise_level:.2f} => Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")
    
    return results

# noise levels
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Evaluate the model on the test set with various noise levels
results = evaluate_with_noise(best_model, X_test_scaled, y_test_scaled, noise_levels)

# Function to simulate sudden increase and decrease in traffic flow
def simulate_traffic_changes(data, increase_idx, decrease_idx, increase_factor=1.5, decrease_factor=0.5):
    data_copy = data.copy()
    
    # Simulate sudden increase (at 'increase_idx' index)
    data_copy[increase_idx] = data_copy[increase_idx] * increase_factor
    
    # Simulate sudden decrease (at 'decrease_idx' index)
    data_copy[decrease_idx] = data_copy[decrease_idx] * decrease_factor
    
    return data_copy

# Indices for increase and decrease
increase_idx = [50, 100, 150]  # Choose an index for the increase
decrease_idx = [75, 175, 200]  # Choose an index for the decrease

# Simulate the traffic data with sudden changes
noisy_traffic_data = simulate_traffic_changes(X_test_scaled, increase_idx, decrease_idx)

# Evaluate the model before and after the changes
mae_before, rmse_before = evaluate_model(best_model, X_test_scaled, y_test_scaled)
mae_after, rmse_after = evaluate_model(best_model, noisy_traffic_data, y_test_scaled)
