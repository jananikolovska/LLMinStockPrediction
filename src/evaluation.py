import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(actual, predicted):
    """
    Calculate comprehensive metrics for returns prediction evaluation
    
    Args:
        actual: Array of actual return values
        predicted: Array of predicted return values
        
    Returns:
        Dictionary of performance metrics
    """
    # Convert to numpy arrays for consistent handling
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Basic regression metrics
    rmse_val = np.sqrt(mean_squared_error(actual, predicted))
    mae_val = mean_absolute_error(actual, predicted)
    
    # R-squared
    r2_val = r2_score(actual, predicted)
    
    # Directional accuracy (very important for returns)
    actual_direction = np.sign(actual)
    predicted_direction = np.sign(predicted)
    directional_accuracy = np.mean(actual_direction == predicted_direction)
    
    # Hit rate (percentage of predictions in correct direction)
    hit_rate = directional_accuracy  # Same as directional accuracy
    
    # Sharpe-like ratio for predictions (assuming risk-free rate = 0)
    if np.std(predicted) > 0:
        pred_sharpe = np.mean(predicted) / np.std(predicted)
    else:
        pred_sharpe = 0
    
    # Actual Sharpe ratio
    if np.std(actual) > 0:
        actual_sharpe = np.mean(actual) / np.std(actual)
    else:
        actual_sharpe = 0
    
    return {
        'RMSE': rmse_val,
        'MAE': mae_val,
        'R2': r2_val,
        'Directional_Accuracy': directional_accuracy,
        'Hit_Rate': hit_rate,
        'Predicted_Sharpe': pred_sharpe,
        'Actual_Sharpe': actual_sharpe
    }

    
