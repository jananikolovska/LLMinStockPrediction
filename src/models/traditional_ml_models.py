import matplotlib as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def create_lag_features(series, lag_size):
    features = {"value": series}
    for lag in range(1, lag_size + 1):
        features[f"lag_{lag}"] = series.shift(lag)
    return pd.DataFrame(features).dropna()


def split_lag_dataset_to_label_and_features(splits):
    Xs_train = []
    Xs_test = []
    ys_train = []
    ys_test = []
    for lag_dataset in splits:
        train, test = lag_dataset
        Xs_train.append(train.drop('value', axis=1))
        ys_train.append(train['value'])
        Xs_test.append(test.drop('value', axis=1))
        ys_test.append(test['value'])
    return Xs_train, ys_train, Xs_test, ys_test

def update_lag_input(old_input, pred, lag_size):
    new_input = {}
    for i in range(1, lag_size + 1):
        if i == 1:
            new_input[f'lag_1'] = pred
        else:
            new_input[f'lag_{i}'] = old_input.iloc[0][f'lag_{i-1}']
    return pd.DataFrame([new_input])

def recursive_forecast(model, initial_lags, steps):
    """
    Predict future values step-by-step using previous predictions as new inputs.
    
    model: trained regressor
    initial_lags: DataFrame or np.array of lag values, length = number of lags
    steps: number of steps to predict
    """
    preds = []
    pred = np.nan
    for i in range(steps):
        if i < initial_lags.shape[0]:
            x_input = initial_lags.iloc[[i]].copy()
        else:
            x_input = update_lag_input(x_input, pred, len(list(x_input.columns)))
        
        pred = model.predict(x_input)
        preds.append(pred[0])
    return preds

def predict_linear_regression(train, test, lag):
    lr_open_recursive = LinearRegression()
    lr_open_recursive.fit(train[0], train[1])
    initial_open_lags_list = train[0].iloc[-lag:].values.tolist()
    initial_open_lags = pd.DataFrame(initial_open_lags_list, columns=train[0].columns).astype(float)
    
    return recursive_forecast(lr_open_recursive, initial_open_lags, steps=test[1].shape[0])