import tensorflow as tf
import pandas as pd
import numpy as np
from src.train import train_model
from src.evaluate import evaluate_model, model
from src.preprocessing import load_data
from src.model import transformer

def main():
    # Train the model
    train_model()
    
    # Evaluate the model on the test set
    mae, rmse = evaluate_model(model, X_test_scaled, y_test_scaled)
    print(f"Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")
    
if __name__ == "__main__":
    _, _, _, _, X_test_scaled, y_test_scaled = load_data()
    main()
    
