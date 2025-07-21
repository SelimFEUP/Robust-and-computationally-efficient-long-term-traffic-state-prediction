import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, LayerNormalization, Activation, SpatialDropout1D, Add
from src.model import transformer
from src.preprocessing import num_features, load_data

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Initialize the model # 3 & 2 best
epoch = 120
d_model = num_features
num_heads = 3
num_layers = 2
units = 160
dropout = 0.1
learning_rate = 0.0005
output_size = num_features
time_steps = 24

def custom_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return mse + 0.5 * mae
    
# lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=1e-3,first_decay_steps=1000,alpha=1e-4)

def train_model():
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled = load_data()
    
    # Build the model
    model = transformer(time_steps,d_model=d_model,num_heads=num_heads,num_layers=num_layers,units=units,dropout=dropout,output_size=output_size)
    # Compile the model
    model.compile(loss=custom_loss,optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=7, mode='min')
    mc = tf.keras.callbacks.ModelCheckpoint('models/model.keras', monitor='val_mean_absolute_error', verbose=7, save_best_only=True, mode='min') #, save_weights_only=True)
    history = model.fit(X_train_scaled, y_train_scaled, validation_data=(X_val_scaled, y_val_scaled), epochs=epoch, batch_size=32,callbacks=[mc, early_stopping])
    
    # Save training history
    #df_h = pd.DataFrame.from_dict(history.history)
    #df_h.to_csv('models/model_his.csv')
    return history
