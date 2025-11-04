import matplotlib.pyplot as plt

def plot_predictions_vs_actual(results_dict, forecast_horizon=30, figsize=(15, 12), idx=0):
    """
    Plot predicted vs actual returns for all models with subplots
    
    Args:
        results_dict: Dictionary containing model results
        forecast_horizon: Number of forecast periods
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Predicted vs Actual Returns Comparison', fontsize=16, fontweight='bold')
    
    models = ['arima', 'gpt3', 'gpt4']
    model_names = ['ARIMA', 'GPT-3.5', 'GPT-4']
    colors = ['blue', 'red', 'green']
    
    # Time axis for plotting
    time_axis = range(1, forecast_horizon + 1)
    
    # Individual model plots
    for i, (model_key, model_name, color) in enumerate(zip(models, model_names, colors)):
        if model_key in results_dict and results_dict[model_key] is not None:
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            actual = results_dict[model_key]['actual'][idx]
            predicted = results_dict[model_key]['predictions'][idx]
            
            ax.plot(time_axis, actual, 'o-', color='black', label='Actual', alpha=0.7, markersize=4)
            ax.plot(time_axis, predicted, 's--', color=color, label=f'{model_name} Predicted', alpha=0.8, markersize=4)
            
            # Calculate and display metrics
            r2 = results_dict[model_key]['statistical_metrics']['R2']
            dir_acc = results_dict[model_key]['statistical_metrics']['Directional_Accuracy']
            
            ax.set_title(f'{model_name} Predictions\nR² = {r2:.3f}, Dir. Acc. = {dir_acc:.1%}')
            ax.set_xlabel('Forecast Period')
            ax.set_ylabel('Returns')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Combined comparison plot
    ax_combined = axes[1, 1]
    ax_combined.clear()
    
    # Plot actual values once
    if any(results_dict[key] is not None for key in models if key in results_dict):
        # Use actual values from any available model (they should be the same)
        actual_values = None
        for model_key in models:
            if model_key in results_dict and results_dict[model_key] is not None:
                actual_values = results_dict[model_key]['actual'][idx]
                break
        
        if actual_values is not None:
            ax_combined.plot(time_axis, actual_values, 'o-', color='black', 
                           label='Actual', linewidth=2, markersize=5, alpha=0.8)
            
            # Plot predictions for each model
            for model_key, model_name, color in zip(models, model_names, colors):
                if model_key in results_dict and results_dict[model_key] is not None:
                    predicted = results_dict[model_key]['predictions'][idx]
                    ax_combined.plot(time_axis, predicted, '--', color=color, 
                                   label=f'{model_name}', linewidth=1.5, alpha=0.7)
    
    ax_combined.set_title('All Models Comparison')
    ax_combined.set_xlabel('Forecast Period')
    ax_combined.set_ylabel('Returns')
    ax_combined.legend()
    ax_combined.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_cumulative_returns_simulation(results_dict, initial_capital=10000, figsize=(12, 8), idx = 0):
    """
    Plot cumulative returns over time for trading simulation
    
    Args:
        results_dict: Dictionary containing model results
        initial_capital: Starting capital amount
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    models = ['arima', 'gpt3', 'gpt4']
    model_names = ['ARIMA', 'GPT-3.5', 'GPT-4']
    colors = ['blue', 'red', 'green']
    
    # Create time axis
    max_periods = 0
    for model_key in models:
        if model_key in results_dict and results_dict[model_key] is not None:
            max_periods = max(max_periods, len(results_dict[model_key]['actual'][idx]))
    
    time_axis = range(1, max_periods + 1)
    
    # Plot buy-and-hold strategy (baseline)
    if any(results_dict[key] is not None for key in models if key in results_dict):
        # Get actual returns from any available model
        actual_returns = None
        for model_key in models:
            if model_key in results_dict and results_dict[model_key] is not None:
                actual_returns = results_dict[model_key]['actual'][idx]
                break
        
        if actual_returns is not None:
            # Calculate buy-and-hold cumulative returns
            buy_hold_cumulative = [initial_capital]
            capital = initial_capital
            for ret in actual_returns:
                capital = capital * (1 + ret)  # Assuming full investment each period
                buy_hold_cumulative.append(capital)
            
            plt.plot(range(len(buy_hold_cumulative)), buy_hold_cumulative, 
                    '--', color='gray', label='Buy & Hold', linewidth=2, alpha=0.8)
    
    # Plot each model's trading performance
    for model_key, model_name, color in zip(models, model_names, colors):
        if model_key in results_dict and results_dict[model_key] is not None:
            # Simulate cumulative trading returns
            predicted_returns = results_dict[model_key]['predictions'][idx]
            actual_returns = results_dict[model_key]['actual'][idx]
            
            # Simple simulation: invest when predicted return > 0
            cumulative_capital = [initial_capital]
            capital = initial_capital
            
            for pred_ret, actual_ret in zip(predicted_returns, actual_returns):
                if pred_ret > 0:  # Buy signal
                    capital = capital * (1 + actual_ret)
                else:  # Hold cash (assume 0% return)
                    pass  # Capital stays same
                cumulative_capital.append(capital)
            
            plt.plot(range(len(cumulative_capital)), cumulative_capital, 
                    '-', color=color, label=f'{model_name} Trading', linewidth=2)
    
    plt.title('Cumulative Trading Performance Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Period')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add starting capital line
    plt.axhline(y=initial_capital, color='black', linestyle=':', alpha=0.5, label='Initial Capital')
    
    plt.tight_layout()
    plt.show()


def create_performance_summary_table(results_dict):
    """
    Create a comprehensive performance summary table
    
    Args:
        results_dict: Dictionary containing model results
    """
    import pandas as pd
    
    models = ['arima', 'gpt3', 'gpt4']
    model_names = ['ARIMA', 'GPT-3.5', 'GPT-4']
    
    summary_data = []
    
    for model_key, model_name in zip(models, model_names):
        if model_key in results_dict and results_dict[model_key] is not None:
            stats = results_dict[model_key]['statistical_metrics']
            by_strategy = results_dict[model_key]['by_strategy']
            
            # Find best strategy by return
            best_strategy = max(by_strategy.keys(), 
                              key=lambda s: by_strategy[s]['summary']['avg_total_return'])
            best_metrics = by_strategy[best_strategy]['summary']
            
            summary_data.append({
                'Model': model_name,
                'RMSE': f"{stats['RMSE']:.6f}",
                'MAE': f"{stats['MAE']:.6f}",
                'R²': f"{stats['R2']:.4f}",
                'Directional Accuracy': f"{stats['Directional_Accuracy']:.2%}",
                'Total Return': f"{best_metrics['avg_total_return']:.2%}",
                'Final Capital': f"${best_metrics['avg_final_capital']:.2f}",
                'Profitable Trades': f"{best_metrics['avg_profitable_pct']:.2%}",
                'Total Trades': f"{best_metrics['avg_num_trades']:.1f}",
                'Best Strategy': best_strategy
            })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        print("\n" + "="*80)
        print("COMPREHENSIVE PERFORMANCE SUMMARY TABLE")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        return df
    else:
        print("No valid model data available for summary table")
        return None