import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, SpatialDropout1D, Add, Dense, LayerNormalization
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import random

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Load the dataset
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
def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data[i:(i + input_steps), :])
        y.append(data[(i + input_steps):(i + input_steps + output_steps), :])
    return np.array(X), np.array(y)

# Split the data into training, validation, and testing sets
def split_data(X, y, test_size=0.2, val_size=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size/(test_size + val_size))
    return X_train, X_val, X_test, y_train, y_val, y_test


# Load the dataset
filepath = 'transformed_data.csv'  # Update to your dataset path
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

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.1, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads
        self.dropout_rate = dropout
        
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)
        self.dense = tf.keras.layers.Dense(units=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        
        if mask is not None:
            logits += (mask * -1e9)
            
        attention_weights = tf.nn.softmax(logits, axis=-1)
        attention_weights = self.dropout(attention_weights)
        return tf.matmul(attention_weights, value)

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        scaled_attention = self.scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        return self.dense(concat_attention)

class SpatioTemporalPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, num_sensors, d_model, time_steps):
        super(SpatioTemporalPositionalEncoding, self).__init__()
        self.num_sensors = num_sensors
        self.d_model = d_model
        self.time_steps = time_steps

        # Learnable spatial embeddings
        self.spatial_embedding = self.add_weight(
            shape=(num_sensors, d_model),
            initializer="random_normal",
            trainable=True,
            name="spatial_embedding"
        )

        # Precompute temporal encoding
        self.temporal_encoding = self.compute_temporal_encoding()

    def compute_temporal_encoding(self):
        position = np.arange(self.time_steps)[:, np.newaxis]  # (time_steps, 1)
        angle_rates = 1 / np.power(10000, (2 * (np.arange(self.d_model)//2) / np.float32(self.d_model)))
        angle_rates = angle_rates[np.newaxis, :]  # (1, d_model)
        
        angle_rads = position * angle_rates  # (time_steps, d_model)
        
        # Apply sin to even indices, cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        return tf.constant(angle_rads, dtype=tf.float32)

    def call(self, x):
        if isinstance(x, tf.sparse.SparseTensor):
            x = tf.sparse.to_dense(x)

        batch_size = tf.shape(x)[0]
        # Expand dimensions for broadcasting
        spatial = self.spatial_embedding[tf.newaxis, :, :]  # (1, num_sensors, d_model)
        temporal = self.temporal_encoding[:, tf.newaxis, :]  # (time_steps, 1, d_model)
        
        # Combine spatial and temporal encodings
        encoding = spatial + temporal  # (time_steps, num_sensors, d_model)
        encoding = tf.reduce_sum(encoding, axis=1)  # (time_steps, d_model)
        encoding = tf.broadcast_to(encoding, [batch_size, self.time_steps, self.d_model])
        
        return x + encoding

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.time_steps, self.d_model)

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

def custom_loss(y_true, y_pred):
    # For TensorFlow 2.19+ (new API)
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

# Complete Model
def transformer(time_steps, d_model, num_heads, num_layers, units, dropout, output_size):
    input_shape = (time_steps, num_features)
    inputs = tf.keras.Input(shape=input_shape, name="inputs")
    padding_mask = tf.keras.layers.Lambda(
        lambda x: create_padding_mask(tf.reduce_sum(x, axis=-1))
    )(inputs)
    
    dynamic_features = DynamicFeatureEmbedding(input_dim=time_steps, embedding_dim=d_model)(inputs)
    x = tf.keras.layers.Concatenate(axis=-1)([inputs, dynamic_features])
    
    # Projection with learnable scaling factor
    projection = DeepProjectionLayer(d_model=d_model, num_hidden=d_model * 2)(x)
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

# Initialize the model
d_model = num_features
num_heads = 3
num_layers = 2
units = 160
dropout = 0.1
learning_rate = 0.0005
output_size = num_features
time_steps = 24


# Build the model
best_model = transformer(time_steps,d_model=d_model,num_heads=num_heads,num_layers=num_layers,units=units,dropout=dropout,output_size=output_size)

# Compile the model
best_model.compile(loss=custom_loss,optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),metrics=[tf.keras.metrics.MeanAbsoluteError()])
#best_model.summary()

# Load model
best_model.load_weights('model_pems_24.keras')

# Adversarial Test
def adversarial_test(model, X_test, y_test, epsilon=0.10):
    """
    Tests model robustness against adversarial perturbations.
    
    Args:
        model: Trained Keras model
        X_test: Test features (shape: [samples, timesteps, features])
        y_test: Corresponding test labels (shape: [samples, timesteps, features])
        epsilon: Perturbation magnitude
    """
    # Convert to tensors
    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)
    
    # Ensure we only use the matching samples
    n_samples = min(len(X_test), len(y_test))
    X_test_tensor = X_test_tensor[:n_samples]
    y_test_tensor = y_test_tensor[:n_samples]
    
    with tf.GradientTape() as tape:
        tape.watch(X_test_tensor)
        pred = model(X_test_tensor)
        
        # Ensure predictions and labels have same shape
        pred = pred[:, :y_test_tensor.shape[1], :y_test_tensor.shape[2]]
        
        # Calculate loss
        loss = tf.keras.losses.MeanSquaredError()(
            tf.reshape(y_test_tensor, [-1, y_test_tensor.shape[-1]]),
            tf.reshape(pred, [-1, pred.shape[-1]])
        )
    
    # Calculate gradient
    grad = tape.gradient(loss, X_test_tensor)
    
    # Apply perturbation
    X_perturbed = X_test_tensor + epsilon * tf.sign(grad)
    
    # Evaluate
    original_mae = tf.keras.metrics.MeanSquaredError()(
        y_test_tensor, 
        model.predict(X_test_tensor)
    ).numpy().mean()
    
    perturbed_mae = tf.keras.metrics.MeanSquaredError()(
        y_test_tensor,
        model.predict(X_perturbed)
    ).numpy().mean()
    
    original_rmse = tf.keras.metrics.RootMeanSquaredError()(
        y_test_tensor, 
        model.predict(X_test_tensor)
    ).numpy().mean()
    
    perturbed_rmse = tf.keras.metrics.RootMeanSquaredError()(
        y_test_tensor,
        model.predict(X_perturbed)
    ).numpy().mean()
    
    print(f"Original MAE: {original_mae:.4f}")
    print(f"Perturbed MAE (ε={epsilon}): {perturbed_mae:.4f}")
    print(f"MAE change: {perturbed_mae - original_mae:.4f}")
    print(f"Original RMSE: {original_rmse:.4f}")
    print(f"Perturbed RMSE (ε={epsilon}): {perturbed_rmse:.4f}")
    print(f"RMSE change: {perturbed_rmse - original_rmse:.4f}")
 
adversarial_test(best_model, X_test_scaled, y_test_scaled, epsilon=0.05) # epsilon=0.10, 0.20, 0.30
