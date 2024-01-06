import pandas as pd
import numpy as np

def replaceNAN(data):
    if isinstance(data, pd.DataFrame):
        means = data.mean(numeric_only=True)  # Calculate means for each column

        for column in data.columns:
            nan_locs = data[column].isna()
            if nan_locs.any():
                data.loc[nan_locs, column] = means[column]
    elif isinstance(data, np.ndarray):
        means = np.nanmean(a=data, axis=0)  # Automatically detects NaN's and takes them out of the equation
        locs = np.where(np.isnan(data))  # Return locations
        data[locs] = means[locs[1]]

    return data