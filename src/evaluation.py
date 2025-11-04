import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(actual, predicted):
    """
    Calculate metrics for multiple series of predictions.
    If actual and predicted are lists of lists, compute per-series metrics and average them.
    """
    actual = np.array(actual, dtype=object)
    predicted = np.array(predicted, dtype=object)
    
    n_series = len(actual)
    metrics_list = []

    for a, p in zip(actual, predicted):
        a = np.array(a)
        p = np.array(p)

        rmse_val = np.sqrt(mean_squared_error(a, p))
        mae_val = mean_absolute_error(a, p)
        r2_val = r2_score(a, p)

        actual_direction = np.sign(a)
        predicted_direction = np.sign(p)
        directional_accuracy = np.mean(actual_direction == predicted_direction)

        # Sharpe ratios
        pred_sharpe = np.mean(p) / np.std(p) if np.std(p) > 0 else 0
        actual_sharpe = np.mean(a) / np.std(a) if np.std(a) > 0 else 0

        metrics_list.append({
            'RMSE': rmse_val,
            'MAE': mae_val,
            'R2': r2_val,
            'Directional_Accuracy': directional_accuracy,
            'Predicted_Sharpe': pred_sharpe,
            'Actual_Sharpe': actual_sharpe
        })

    # Average across series
    avg_metrics = {
        k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]
    }

    return avg_metrics


    
