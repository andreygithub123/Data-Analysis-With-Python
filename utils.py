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


# standarization - the final matrix to have mean 0 and sd 1 on each column
# standard deviation = average distance from the average
# standarization - how far the element is from the average of its own variable type
# [ ni - n(mean) ] / sd ; ni-X matrix ; n(mean)-X(mean) for each column on the matrix; sd-sd for each column on the matrix
def standardize(X): # assume that we recieve numpy.ndarray
    means = np.mean(a=X, axis=0) #axis 0 - compute the means on the columns
    stds = np.std(a=X,axis=0)
    return (X-means) / stds