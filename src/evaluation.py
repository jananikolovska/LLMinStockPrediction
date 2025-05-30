import numpy as np
import pandas as pd

# Mean Absolute Percentage Error (MAPE)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def gain(C, C_pred, O, O_pred):
    """
    Function adapted from: DISI - Università di Bologna, Cesena
    Original author(s): Prof. Gianluca Moro, Roberto Pasolini et al.
    Adapted by: Jana Nikolovska
    """
    C_pred = pd.Series(C_pred)
    O_pred = pd.Series(O_pred)
    # Calculate the differences between actual and predicted open/close prices
    CO_diff = C - O  # Difference between actual close and actual open
    
    # Compare predicted close with actual open (whether trade will grow or decline)
    growth = C_pred > O_pred
    decline = C_pred < O_pred
    growth.index = CO_diff.index
    decline.index = CO_diff.index
 
    # Calculate the gain (positive difference in growth, negative in decline)
    return CO_diff[growth].sum() - CO_diff[decline].sum()

def roi(C, C_pred, O, O_pred):
    """
    Function adapted from: DISI - Università di Bologna, Cesena
    Original author(s): Prof. Gianluca Moro, Roberto Pasolini et al.
    Adapted by: Jana Nikolovska
    """
    # Calculate mean of the actual open prices
    mean_open = O.mean()  
    # Calculate ROI based on gain and mean open price
    return gain(C, C_pred, O, O_pred) / mean_open

def print_eval(preds_open, preds_close, y_open, y_close, verbose=True):
    """
    Function adapted from: DISI - Università di Bologna, Cesena
    Original author(s): Prof. Gianluca Moro, Roberto Pasolini et al.
    Adapted by: Jana Nikolovska
    """
    # preds_open: Predicted open values
    # preds_close: Predicted close values
    # y_open: Actual open values
    # y_close: Actual close values

    # Print Gain and ROI based on both open and close predictions
    gain_ = gain(y_close, preds_close, y_open, preds_open)
    roi_ = roi(y_close, preds_close, y_open, preds_open)
    if verbose:
        print("Gain: {:.2f}$".format(gain_))
        print("ROI: {:.3%}".format(roi_))
    return gain_, roi_
