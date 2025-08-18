
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rmse(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)

def r2(y_true, y_pred):
    import numpy as np
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

def time_train_test_split(df, date_col, test_days):
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = len(df) - int(test_days)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def plot_series(dates, y, yhat=None, title=""):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.plot(dates, y, label="actual")
    if yhat is not None:
        plt.plot(dates, yhat, label="pred")
    plt.title(title); plt.xlabel("date"); plt.ylabel("value")
    plt.legend(); plt.tight_layout(); plt.show()

def seasonal_features(dates):
    import numpy as np, pandas as pd
    day = pd.to_datetime(dates).dayofyear.values
    s1 = np.sin(2*np.pi*day/365.25)
    c1 = np.cos(2*np.pi*day/365.25)
    return np.vstack([s1, c1]).T
