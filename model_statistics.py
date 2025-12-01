import numpy as np
  # Ordinary Least Squares (OLS)
def OrdinaryLeastSquares(x, y):
    n = len(x)

    numerator = n*np.sum(x*y) - np.sum(x) * np.sum(y)
    denominator = n*np.sum(x**2) - np.sum(x)**2
    m = numerator / denominator

    b = np.mean(y) - m * np.mean(x)

    return m,b

def correlation(x, y):
    n = len(x)

    numerator = n*np.sum(x*y) - np.sum(x) * np.sum(y)
    denominator = (n*np.sum(x**2) - np.sum(x)**2) * (n * np.sum(y**2) - np.sum(y)**2)

    r = numerator / np.sqrt(denominator)

    return r

# Mean Absolute Error (MAE)
def MeanAbsError(predicted, real):
    return np.sum(np.abs(predicted - real)) / len(real)

# Root Mean Squared Error (RMSE)
def RMSE(predicted, real):
    return np.sum(np.square(predicted - real)) / len(real)