import numpy as np

def outlier_thresholds_normal(dataframe, cols, z_threshold=3):
        df = dataframe.copy()
        thresholds = {}
        for col in cols:
            # Upper and lower bounds
            upper_limit = df[col].mean() + z_threshold*df[col].std()
            lower_limit = df[col].mean() - z_threshold*df[col].std()
            thresholds[col] = [upper_limit, lower_limit]
        return thresholds


def outlier_thresholds_skewed(dataframe, cols, iqr_threshold=1.5, upper_limit=None, lower_limit=None):
    df = dataframe.copy()
    thresholds = {}
    for col in cols:
        # Upper and lower bounds
        if upper_limit == None:
            ul = df[col].quantile(0.75) + iqr_threshold*(df[col].quantile(0.75)-df[col].quantile(0.25))
        else:
            ul = upper_limit
        if lower_limit == None:
            ll = df[col].quantile(0.25) - iqr_threshold*(df[col].quantile(0.75)-df[col].quantile(0.25))
        else:
            ll = lower_limit
        thresholds[col] = [ul, ll]
    return thresholds

def apply_thresholds(dataframe, cols, thresholds):
    df = dataframe.copy()
    for col in cols:
        # Upper and lower bounds
        upper_limit = thresholds[col][0]
        lower_limit = thresholds[col][1]
        # Ammend value if above the upper limit.
        # np.where parameters are (condition, value if true, value if false)
        df[col] = np.where(df[col]>upper_limit, upper_limit, df[col])
        # Ammend value if below the lower limit
        df[col] = np.where(df[col]<lower_limit, lower_limit, df[col])
    return df

# Create an outlier function that sets negative values to 0
def set_negativestozero(df, cols):
    for col in cols:
        df[col] = df[col].apply(lambda x: 0 if x < 0 else x)
    return df

# Create an outlier function that sets upper and lower bounds. The input is the original dataframe and a dictionary with the
# column names as keys and the upper and lower bounds as values.
def set_bounds(df, bounds):
    for col, bound in bounds.items():
        df[col] = df[col].apply(lambda x: bound[0] if x < bound[0] else x)
        df[col] = df[col].apply(lambda x: bound[1] if x > bound[1] else x)
    return df

# Function for setting upper and lower bounds based on z-scores.
def bounds_zscore(df, cols, zscore=3):
    for col in cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        df[col] = df[col].apply(lambda x: col_mean - zscore*col_std if x < col_mean - zscore*col_std else x)
        df[col] = df[col].apply(lambda x: col_mean + zscore*col_std if x > col_mean + zscore*col_std else x)
    return df

# Function for setting upper and lower bounds based on IQR.
def bounds_iqr(df, cols, k=1.5):
    for col in cols:
        col_q1 = df[col].quantile(0.25)
        col_q3 = df[col].quantile(0.75)
        col_iqr = col_q3 - col_q1
        df[col] = df[col].apply(lambda x: col_q1 - k*col_iqr if x < col_q1 - k*col_iqr else x)
        df[col] = df[col].apply(lambda x: col_q3 + k*col_iqr if x > col_q3 + k*col_iqr else x)
    return df

# Function to apply log transformation to a column. Add a small value to avoid taking the log of 0.
def log_transform(df, cols, const=1):
    for col in cols:
        df[col] = df[col].apply(lambda x: np.log(x+const))
    return df

# Function to apply square root transformation to a column.
def sqrt_transform(df, cols):
    for col in cols:
        df[col] = df[col].apply(lambda x: np.sqrt(x))
    return df
