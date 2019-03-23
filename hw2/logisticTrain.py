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

# --- hyperparameters ---
#np.random.seed(138049712)
np.random.seed(880301)

batchSize = 32 # 64, 128, 256 ...
epoch = 50

# Adam
lr = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

customLambda = 0.001
candidateLambdas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
crossValidationK = 10
crossValidationEnabled = False

np.warnings.filterwarnings('ignore')
# --- hyperparameters ---

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def loss(w, X, y, lb):
    fX = sigmoid(X @ w)
    return -np.sum(y * np.log(fX + epsilon) + (1 - y) * np.log(1 + epsilon - fX)) + lb / 2.0 * np.sum(w[1:, :] ** 2)

def gradient(w, X, y, lb):
    grad = X.T @ (sigmoid(X @ w) - y)
    grad[1:, :] += lb * w[1:, :]
    return grad

def accuracy(w, X, y):
    return np.sum((sigmoid(X @ w) >= 0.5) == y) / X.shape[0]

def logisticRegression(X, y, lb):
    #print('Logistic Regression')
    #print('Epoch:', epoch)
    #print('Lambda:', lb)
    #print('Number of Data:', X.shape[0])
    #print('Number of Features:', X.shape[1])
    w = np.random.rand(X.shape[1], 1)
    m = np.zeros((X.shape[1], 1))
    v = np.zeros((X.shape[1], 1))
    print('Initial State; Loss: %10.8f; Accuracy: %10.8f' % (loss(w, X, y, lb), accuracy(w, X, y)))
    for ep in range(epoch):
        for t in range((X.shape[0] + batchSize - 1) // batchSize):
            bX = X[t * batchSize:(t+1) * batchSize, :]
            by = y[t * batchSize:(t+1) * batchSize, :]
            grad = gradient(w, bX, by, lb)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            mhat = m / (1 - beta1 ** (t+1))
            vhat = v / (1 - beta2 ** (t+1))
            w -= lr * mhat / (np.sqrt(vhat) + epsilon)
        print('Epoch %7d; Loss: %10.8f; Accuracy: %10.8f' % (ep+1, loss(w, X, y, lb), accuracy(w, X, y)))
    print('Final   State; Loss: %10.8f; Accuracy: %10.8f' % (loss(w, X, y, lb), accuracy(w, X, y)))
    return w

print('Reading data')
X = loadX('./X_train', 'training')
y = loadY('./Y_train')

if crossValidationEnabled:
    print('Start ' + str(crossValidationK) + '-fold cross validation')
    Data = np.hstack([X, y])
    np.random.shuffle(Data)
    Data = np.array_split(Data, crossValidationK)

    bestLambda, bestEcvAvg = 0.0, np.inf
    for lb in candidateLambdas:
        EinSum, EcvSum = 0.0, 0.0
        for i in range(crossValidationK):
            tData = np.vstack(Data[0:i] + Data[i+1:])
            vData = Data[i]
            tX, ty = tData[:, :-1], tData[:, -1:]
            vX, vy = vData[:, :-1], vData[:, -1:]

            w = logisticRegression(tX, ty, lb)
            EinSum += 1.0 - accuracy(w, tX, ty)
            EcvSum += 1.0 - accuracy(w, vX, vy)
        EinAvg = EinSum / crossValidationK
        EcvAvg = EcvSum / crossValidationK
        print('lambda: %10f; Average Ein: %10.8f; Average Ecv: %10.8f' % (lb, EinAvg, EcvAvg))
        if EcvAvg < bestEcvAvg:
            bestLambda, bestEcvAvg = lb, EcvAvg

    print('Best Lambda: %10f; Best Ecv: %10.8f' % (bestLambda, bestEcvAvg))
    print('Start final training')
    tData = np.vstack(Data)
    tX, ty = tData[:, :-1], tData[:, -1:]
    w = logisticRegression(tX, ty, bestLambda)
else:
    w = logisticRegression(X, y, customLambda)

print('Saving model')
np.save('modelL', w)
print('Done')
