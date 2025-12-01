import numpy as np

def bootstrap_sample(X, y):
    n = len(X)
    idx = np.random.choice(n, n, replace=True)
    return X[idx], y[idx]
