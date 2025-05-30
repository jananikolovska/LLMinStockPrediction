import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
import random
import os
from dash import Dash, dcc, html, Input, Output

RESULT_IMAGES_PATH = 'visualizations/project'

def random_8_digit_number():
    return random.randint(10_000_000, 99_999_999)

def plot_open_close(df):
    """Plot Open and Close prices over time."""
    df[['Open', 'Close']].plot(figsize=(14, 6), title='Stock Open and Close Prices')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

def plot_daily_returns(df):
    """Plot daily returns and their distribution."""
    df['Daily Return'].plot(figsize=(12, 4), title='Daily Returns')
    plt.grid(True)
    plt.show()

    sns.histplot(df['Daily Return'].dropna(), kde=True, bins=50)
    plt.title('Distribution of Daily Returns')
    plt.show()

def plot_rolling_stats(df):
    """Plot rolling mean and std deviation."""
    df[['Close', 'Rolling Mean', 'Rolling Std']].plot(figsize=(14, 6))
    plt.title('Rolling Mean and STD (20 days)')
    plt.grid(True)
    plt.show()

def plot_correlation(df):
    """Scatter plot to visualize correlation between Open and Close."""
    corr = df['Open'].corr(df['Close'])
    print(f"Correlation between Open and Close: {corr:.2f}")

    sns.scatterplot(x='Open', y='Close', data=df)
    plt.title('Open vs Close Price')
    plt.grid(True)
    plt.show()

def plot_autocorrelation(df):
    """Plot autocorrelation of Close prices."""
    autocorrelation_plot(df['Close'].dropna())
    plt.title('Autocorrelation of Close Prices')
    plt.show()

def plot_seasonality(df):
    """Plot weekly and monthly average prices."""
    df['Weekday'] = df.index.weekday
    weekly_avg = df.groupby('Weekday')[['Open', 'Close']].mean()
    weekly_avg.plot(kind='bar', figsize=(10, 5), title='Average Prices by Weekday')
    plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
    plt.grid(True)
    plt.show()

    monthly_avg = df.resample('M').mean()
    monthly_avg[['Open', 'Close']].plot(figsize=(14, 6), title='Monthly Average Prices')
    plt.grid(True)
    plt.show()
    
def get_volatility_returns_figure(open_, close_, sliding_window_size):
    daily_returns_close = close_.pct_change()
    daily_returns_open = open_.pct_change()

    volatility_open = daily_returns_open.rolling(window=sliding_window_size).std()
    volatility_close = daily_returns_close.rolling(window=sliding_window_size).std()

    fig = go.Figure(make_subplots(rows=2, cols=2, 
                                  subplot_titles=[
                                      'Histogram of Daily Returns (Open)',
                                      'Rolling Volatility (Open)',
                                      'Histogram of Daily Returns (Close)',
                                      'Rolling Volatility (Close)'
                                  ]
                                ))

    fig.add_trace(
        go.Histogram(x=daily_returns_open.dropna(), nbinsx=50, marker_color='blue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=volatility_open.index, y=volatility_open, mode='lines', line=dict(color='red')),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=daily_returns_close.dropna(), nbinsx=50, marker_color='green'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=volatility_close.index, y=volatility_close, mode='lines', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(height=700, width=1000, title_text="Volatility and Returns Analysis", showlegend=False)
    return fig


def get_moving_averages_figure(open_, close_, sliding_window_size):
    sma_close_ = close_.rolling(window=sliding_window_size).mean()
    ema_close_ = close_.ewm(span=sliding_window_size, adjust=False).mean()
    sma_open_ = open_.rolling(window=sliding_window_size).mean()
    ema_open_ = open_.ewm(span=sliding_window_size, adjust=False).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=open_.index, y=open_, mode='lines', name='Open', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=sma_open_.index, y=sma_open_, mode='lines', name=f'SMA Open {sliding_window_size}', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=ema_open_.index, y=ema_open_, mode='lines', name=f'EMA Open {sliding_window_size}', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=close_.index, y=close_, mode='lines', name='Close', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=sma_close_.index, y=sma_close_, mode='lines', name=f'SMA Close {sliding_window_size}', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=ema_close_.index, y=ema_close_, mode='lines', name=f'EMA Close {sliding_window_size}', line=dict(color='brown')))
    fig.update_layout(
        title=f'Moving Averages with Window Size: {sliding_window_size}',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Click to toggle lines',
        hovermode='x unified',
        height=600
    )
    return fig


def get_candlestick_figure(open_, high_, low_, close_, ds_name):
    fig = go.Figure(data=[
        go.Candlestick(
            x=close_.index,
            open=open_,
            high=high_,
            low=low_,
            close=close_,
            name=ds_name
        )
    ])
    fig.update_layout(
        title=f"{ds_name} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )
    return fig


def get_acf_pacf_figure(open_, close_, lags):
    acf_open = acf(open_, nlags=lags, fft=False)
    pacf_open = pacf(open_, nlags=lags)
    acf_close = acf(close_, nlags=lags, fft=False)
    pacf_close = pacf(close_, nlags=lags)

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        f"ACF - Open (lags={lags})",
        f"PACF - Open (lags={lags})",
        f"ACF - Close (lags={lags})",
        f"PACF - Close (lags={lags})"
    ])

    fig.add_trace(go.Bar(x=list(range(len(acf_open))), y=acf_open, name='ACF Open'), row=1, col=1)
    fig.add_trace(go.Bar(x=list(range(len(pacf_open))), y=pacf_open, name='PACF Open'), row=1, col=2)
    fig.add_trace(go.Bar(x=list(range(len(acf_close))), y=acf_close, name='ACF Close'), row=2, col=1)
    fig.add_trace(go.Bar(x=list(range(len(pacf_close))), y=pacf_close, name='PACF Close'), row=2, col=2)

    fig.update_layout(height=700, width=1000, title_text="Autocorrelation Analysis", showlegend=False)
    return fig

def gain_over_tries(gain, model_name='', hash_number=None):
    plt.figure(figsize=(5, 3))
    sns.histplot(gain, bins=8, kde=True, color='skyblue', edgecolor='white', linewidth=1.5)
    plt.title(f"{model_name} Gain Over Tries", fontsize=14)
    plt.xlabel("Value")
    plt.ylabel("Density", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if hash_number:
        filename = os.path.join(RESULT_IMAGES_PATH, f'plot_{hash_number}.png')
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
    
def plot_eval_over_time(preds_open, preds_close, y_open, y_close, hash_number=None):
    # Compute actual daily gain (C - O)
    CO_diff = y_close - y_open

    # Make sure predictions are pandas Series with the same index
    preds_close = pd.Series(preds_close, index=y_close.index)
    preds_open = pd.Series(preds_open, index=y_open.index)

    # Prediction directions
    growth = preds_close > preds_open
    decline = preds_close < preds_open

    # Calculate daily gain
    daily_gain = pd.Series(0, index=CO_diff.index, dtype="float64")
    daily_gain[growth] = CO_diff[growth]
    daily_gain[decline] = -CO_diff[decline]

    # Cumulative gain
    cumulative_gain_series = daily_gain.cumsum()

    # Plot both
    df = pd.DataFrame({
    'date': list(y_open.index),
    'Cumulative Gain': cumulative_gain_series,
    'Daily Gain': daily_gain
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['Cumulative Gain'],
        mode='lines',
        name='Cumulative Gain ($)',
        hovertemplate='Date: %{x}<br>Cumulative Gain: %{y:.2f}<extra></extra>',
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['Daily Gain'],
        name='Daily Gain ($)',
        hovertemplate='Date: %{x}<br>Daily Gain: %{y:.2f}<extra></extra>',
        marker=dict(color='orange'),
        opacity=0.6
    ))
    
    fig.update_layout(
        title='Daily and Cumulative Gain Over Time',
        xaxis_title='Date',
        yaxis_title='Gain ($)',
        legend_title='Legend',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        width=1000
    )

    if hash_number:
        filename = os.path.join(RESULT_IMAGES_PATH, f'plot_{hash_number}.png')
        fig.write_image(filename)
    else:
        fig.show()
    return daily_gain, cumulative_gain_series

def plot_autoregressive_ml_model_results(y_train, y_test, preds, ds_name='', model_name='', hash_number=None):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=y_train.index, y=y_train, mode='lines', name='Historic Data', line=dict(color='blue', width=2)))
    
    fig.add_trace(go.Scatter(x=y_test.index, y=preds, mode='lines', name='Predicted Data', line=dict(color='orange', dash='dash', width=2)))
    
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='True Data', line=dict(color='green', width=2)))
    
    fig.update_layout(
        title=f"Forecasting Plot {ds_name} {model_name}: Predicted vs True Data",
        xaxis_title="Time",
        yaxis_title="Values",
        template="plotly",
        showlegend=True,
    )

    
    if hash_number:
        filename = os.path.join(RESULT_IMAGES_PATH, f'plot_{hash_number}.png')
        fig.write_image(filename)
    else:
        fig.show()

