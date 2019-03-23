from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import pickle
import sys

def loadX(path, dataType):
    X = pd.read_csv(path).to_numpy(dtype=float)

    if dataType == 'training':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        np.save('meanE', mean)
        np.save('stdE', std)
    elif dataType == 'test':
        mean = np.load('meanE.npy')
        std = np.load('stdE.npy')
    else:
        print('Error: Unknown data set type')

    std[std == 0.0] = 1
    X = (X - mean) / std

    return X

def loadY(path):
    y = pd.read_csv(path).to_numpy(dtype=float)
    return y

X = loadX(sys.argv[1], 'test')

with open('modelE', 'rb') as f:
    clf = pickle.load(f)

print('Predicting')
y_pred = clf.predict(X).astype(int)

print('Outputting')
with open(sys.argv[2], 'w') as outputFile:
    outputFile.write('id,label\n')
    for i in range(y_pred.shape[0]):
        outputFile.write(str(i+1) + ',' + str(y_pred[i]) + '\n')

print('Done')
