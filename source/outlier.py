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

# Function to replace outliers with upper and lower limits.
# The bounds parameter is a dictionary, where the keys are the column names and the values are lists of the upper and lower bounds.
def outlier_bounds(dataframe, bounds):
    df = dataframe.copy()
    for col in bounds.keys():
        # Upper and lower bounds
        lower_limit = bounds[col][0]
        upper_limit = bounds[col][1]
        # Ammend value if above the upper limit.
        # np.where parameters are (condition, value if true, value if false)
        df[col] = np.where(df[col]>upper_limit, upper_limit, df[col])
        # Ammend value if below the lower limit
        df[col] = np.where(df[col]<lower_limit, lower_limit, df[col])
    return df

# Function to apply the iqr method to replace outliers with upper and lower limits.
def outlier_iqr(dataframe, cols, iqr_threshold=1.5):
    df = dataframe.copy()
    for col in cols:
        # Upper and lower bounds
        ul = df[col].quantile(0.75) + iqr_threshold*(df[col].quantile(0.75)-df[col].quantile(0.25))
        ll = df[col].quantile(0.25) - iqr_threshold*(df[col].quantile(0.75)-df[col].quantile(0.25))
        # Ammend value if above the upper limit.
        # np.where parameters are (condition, value if true, value if false)
        df[col] = np.where(df[col]>ul, ul, df[col])
        # Ammend value if below the lower limit
        df[col] = np.where(df[col]<ll, ll, df[col])
    return df

# Apply log transformation to skewed data
def log_transform(dataframe, cols):
    df = dataframe.copy()
    for col in cols:
        df[col] = np.log1p(df[col])
    return df