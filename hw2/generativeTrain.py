import numpy as np
import pandas as pd

def loadX(path, dataType=None):
    X = pd.read_csv(path).to_numpy(dtype=float)
    return X
def loadY(path, dataType=None):
    y = pd.read_csv(path).to_numpy(dtype=float)
    return y

X = loadX('./X_train')
y = loadY('./Y_train')

X0 = X[y.flatten() == 0]
X1 = X[y.flatten() == 1]

N0 = X0.shape[0]
N1 = X1.shape[0]

mean0 = np.mean(X0, axis=0).reshape((-1, 1))
cov0 = np.cov(X0.T)
mean1 = np.mean(X1, axis=0).reshape((-1, 1))
cov1 = np.cov(X1.T)

cov = (N0 * cov0 + N1 * cov1) / (N0 + N1)

w = ((mean0 - mean1).T @ np.linalg.inv(cov)).T
b = -0.5 * mean0.T @ np.linalg.inv(cov) @ mean0 + 0.5 * mean1.T @ np.linalg.inv(cov) @ mean1 + np.log(N0 / N1)

np.save('wG', w)
np.save('bG', b)
print('Done')
