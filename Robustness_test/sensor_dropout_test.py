def sensor_dropout_test(model, X_test_scaled, y_test_scaled, p_drop=0.1):
    """
    Test model robustness to random sensor dropout using scaled data.
    
    Args:
        model: Trained model (expects scaled input)
        X_test_scaled: Scaled test features (shape: [samples, timesteps, sensors])
        y_test_scaled: Scaled test labels
        p_drop: Probability of dropping each sensor (0.1 = 10% chance per sensor) change it's values such as 0.2, 0.3,..
        to check MAE and RMSE scores
    """
    # Create a broadcastable mask for sensors (last dimension)
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
    
# model, X_test_scaled, y_test_scaled are the saved model, X_test and y_test of your dataset. you also need to define the model as was done in Adversarial_Perturbations_test.py file. 
sensor_dropout_test(model, X_test_scaled, y_test_scaled)
