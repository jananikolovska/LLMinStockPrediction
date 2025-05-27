import pandas as pd
from statsmodels.tsa.stattools import adfuller



def load_dataset(path, visualize=True):
    data = pd.read_csv(path)

    #Set index as Dataframe
    data['Date'] = pd.to_datetime(data['Date'], utc=True)
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    data.set_index('Date', inplace=True)
    data.index = pd.to_datetime(data.index)
    
    #Resample so we have a fixed frequency
    data = data.resample('D').interpolate()  #Resample the data to a daily frequency, to address the missing data caused by non-trading days (interpolation)
    
    if visualize:
        display(data)
    
    return data

def split_dataset(data, test_size=0.2, ds_name = '', verbose=True):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        test_index = int(len(data) * (1 - test_size))
        
        train = data.iloc[:test_index]
        test = data.iloc[test_index:]
        
    else:
        data = np.array(data)  
        test_index = int(len(data) * (1 - test_size))
        
        # Split the data
        train = data[:test_index]
        test = data[test_index:]

    if verbose:
        print(f'Train {ds_name} length: {len(train)}')
        print(f'Test {ds_name} length: {len(test)}')

    return train, test

def adf_test(series, label=''):
    print(f"\nADF Test on {label}")
    result = adfuller(series, autolag='AIC')
    labels = ['ADF Statistic', 'p-value', '# Lags Used', 'Number of Observations Used']
    for value, name in zip(result[:4], labels):
        print(f"{name} : {value}")
    if result[1] <= 0.05:
        print("=> Series is stationary")
    else:
        print("=> Series is NOT stationary")

