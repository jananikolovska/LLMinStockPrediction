def split_time_series(df, train_size, test_size):
    """
    Splits a time series into multiple overlapping train/test pairs (rolling window approach).
    
    Args:
        df (pd.Series or pd.DataFrame): Time series data.
        train_size (int): Number of samples in each training window.
        test_size (int): Number of samples in each test window.
    
    Returns:
        list of tuples: Each tuple contains (train, test) data for one split.
    """
    splits = []
    start = 0
    total_len = len(df)

    while start + train_size + test_size <= total_len:
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:start + train_size + test_size]
        splits.append((train, test))
        start += test_size  # 🔁 rolling forward by test size for overlapping splits

    return splits
