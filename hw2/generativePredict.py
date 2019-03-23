import numpy as np
import pandas as pd
import sys

def loadX(path, dataType=None):
    X = pd.read_csv(path).to_numpy(dtype=float)
    return X
def loadY(path, dataType=None):
    y = pd.read_csv(path).to_numpy(dtype=float)
    return y

X = loadX(sys.argv[1])
w = np.load('wG.npy')
b = np.load('bG.npy')
y_pred = (1 / (1 + np.exp(-(X @ w + b))) <= 0.5).astype(int)

with open(sys.argv[2], 'w') as outputFile:
    outputFile.write('id,label\n')
    for i in range(y_pred.shape[0]):
        outputFile.write(str(i+1) + ',' + str(y_pred[i, 0]) + '\n')

print('Done')
