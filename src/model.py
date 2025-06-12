import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, LayerNormalization, Activation, SpatialDropout1D, Add
from src.preprocessing import *
from keras.regularizers import l2

# constants
d_model = num_features
num_heads = 3
num_layers = 2
units = 160
dropout = 0.1
learning_rate = 0.0005
output_size = num_features
time_steps = 24

# Attention Components
def scaled_dot_product_attention2(query, key, value, mask=None):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    if mask is not None:
        logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    attention_weights = tf.keras.layers.Dropout(rate=dropout)(attention_weights)
    return tf.matmul(attention_weights, value)

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

# STPE layer
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

# TCN Block
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

# DLP layer
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

# Final Model
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
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            output_size,
            bias_initializer=tf.keras.initializers.Constant(0.0)  # Initialize to neutral
        )
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
