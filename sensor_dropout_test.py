import numpy as np
from src.evaluate import model, evaluate_model
from src.preprocessing import load_data

# p_drop: Probability of dropping each sensor (0.1 = 10% chance per sensor) change it's values such as 0.2, 0.3,.. to check MAE and RMSE scores
def sensor_dropout_test(model, X_test_scaled, y_test_scaled, p_drop=0.1):
    # Create a broadcastable mask for sensors
    mask = np.random.choice(
        [0, 1], 
        size=(1, 1, X_test_scaled.shape[-1]),  # Preserve dims for broadcasting
        p=[p_drop, 1 - p_drop]
    )
    
    # Corrupt data by zeroing out dropped sensors
    X_corrupted = X_test_scaled * mask
    
    # Evaluate on scaled data
    mae, rmse = evaluate_model(model, X_corrupted, y_test_scaled)
    print(f"With {p_drop*100:.0f}% sensor dropout prob: MAE={mae:.4f}, RMSE={rmse:.4f}")
    return mae, rmse
    
# model, X_test_scaled, y_test_scaled are the saved model, X_test and y_test of your dataset 
_, _, _, _, X_test_scaled, y_test_scaled = load_data() 
sensor_dropout_test(model, X_test_scaled, y_test_scaled)
