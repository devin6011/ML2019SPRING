import numpy as np
import pandas as pd
import sys

def loadX(path, dataType):
    # load data
    X = pd.read_csv(path).to_numpy(dtype=float)

    # feature transform
    continuousTerm = [0, 1, 3, 4, 5]
    #X = np.delete(X, 1, axis=1)
    #continuousTerm = [0, 2, 3, 4]
    sq = []
    for i in range(len(continuousTerm)):
        for j in range(i, len(continuousTerm)):
            sq.append((X[:, continuousTerm[i]] * X[:, continuousTerm[j]]).reshape(-1, 1))
    sq = np.hstack(sq)
    contNonCont = []
    for i in continuousTerm:
        for j in range(X.shape[1]):
            if j in continuousTerm:
                continue
            contNonCont.append((X[:, i] * X[:, j]).reshape(-1, 1))
    contNonCont = np.hstack(contNonCont)
    #X = np.hstack([X, sq, X[:, continuousTerm] ** 3, contNonCont])
    X = np.hstack([X, X[:, continuousTerm] ** 2, X[:, continuousTerm] ** 3, np.log(X[:, continuousTerm] + 1e-8), np.sqrt(X[:, continuousTerm] + 1e-8), np.reciprocal(X[:, continuousTerm] + 100)])
    #X = np.hstack([X, X[:, continuousTerm] ** 2, X[:, continuousTerm] ** 3])
    #print(X.shape)

    # feature scaling
    if dataType == 'training':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        np.save('meanL', mean)
        np.save('stdL', std)
    elif dataType == 'test':
        mean = np.load('meanL.npy')
        std = np.load('stdL.npy')
    else:
        print('Error: Unknown data set type')
    # prevent division by zero (which yields NaN)
    std[std == 0.0] = 1
    X = (X - mean) / std

    # add bias
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    return X

def loadY(path):
    # load data
    y = pd.read_csv(path).to_numpy(dtype=float)

    return y

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

print('Loading model')
w = np.load('modelL.npy')

print('Predicting')
X = loadX(sys.argv[1], 'test')
predict = (sigmoid(X @ w) >= 0.5).astype(int)

print('Outputting')
with open(sys.argv[2], 'w') as outputFile:
    outputFile.write('id,label\n')
    for i in range(predict.shape[0]):
        outputFile.write(str(i+1) + ',' + str(predict[i, 0]) + '\n')

print('Done')
