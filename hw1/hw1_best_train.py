import sys
import numpy as np
import pandas as pd

np.random.seed(414823)
epoch = 500000
lossPrintGap = 5000
lr = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
lb = 10.0

# read data
print('Loading data...')
data = pd.read_csv('train.csv', encoding='big5', index_col=[0, 2])
data = data.iloc[:, 1:].replace('NR', 0)
data = data.to_numpy(dtype=float)

temp = [np.empty(0) for i in range(18)]
for i, row in enumerate(data):
    temp[i % 18] = np.append(temp[i % 18], row)
data = np.vstack(temp)

# manufacture training data
trainX = []
trainY = []
for month in range(12):
    for beginIndex in range(month * 480, (month + 1) * 480 - 9):
        x = data[:, beginIndex:beginIndex+9].reshape(1, -1)
        y = data[9, beginIndex + 9]
        for i in range(9):
            if x[0, 9*9 + i] < 0:
                break
        else:
            trainX.append(x)
            trainY.append(y)

trainX = np.vstack(trainX)
trainY = np.vstack(trainY)

trainX = np.hstack([trainX[:, 9:135]])
temp = []
for i in [1, 3, 5, 6, 7, 8, 9, 11, 13]:
    temp.append((trainX[:, i*9+8] ** 2).reshape(-1, 1))
for i in range(6, 9):
    for j in [0, 1, 7, 9]:
        temp.append((trainX[:, 8 * 9 + i] * trainX[:, j*9+i]).reshape(-1, 1))
temp = np.hstack(temp)
trainX = np.hstack([trainX, temp])

# feature scaling
mean = np.mean(trainX, axis=0)
std = np.std(trainX, axis=0)
trainX = (trainX - mean) / std

# add bias terms
trainX = np.hstack([np.ones((trainX.shape[0], 1)), trainX])

# shuffle training data and partition into 10 parts
temp = np.hstack([trainX, trainY])
np.random.shuffle(temp)
temp = np.array_split(temp, 10)

# linear regression (Adam)
def loss(w, trainX, trainY):
    return np.sqrt(np.mean(np.square(np.dot(trainX, w) - trainY)))

print('Start training...')
print('Epoch:', epoch)
print('Initial learning rate:', lr)
print('Initial beta1:', beta1)
print('Initial beta2:', beta2)
print('Lambda:', lb)

# 10-fold cross validation
for l in [1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1e+3, 1e+4, 1e+5]:
    EvalSum = 0.0
    EinSum = 0.0
    for i in range(10):
        trainData = np.concatenate(temp[0:i] + temp[i+1:])
        valData = temp[i]
        trainX = trainData[:, :-1]
        trainY = trainData[:, -1:]
        valX = valData[:, :-1]
        valY = valData[:, -1:]

        tata = l * np.eye(trainX.shape[1])
        w = np.linalg.pinv(trainX.T @ trainX + tata) @ (trainX.T @ trainY)
        EinSum += loss(w, trainX, trainY)
        EvalSum += loss(w, valX, valY)

    EinSum /= 10.0
    EvalSum /= 10.0
    print('%10.10f %10.10f %10.10f' % (l, EinSum, EvalSum))

#sys.exit(0)
trainData = np.concatenate(temp)
trainX = trainData[:, :-1]
trainY = trainData[:, -1:]
w = np.random.rand(trainX.shape[1], 1)
m = np.zeros((trainX.shape[1], 1))
v = np.zeros((trainX.shape[1], 1))
print('Initial Loss:', loss(w, trainX, trainY))

XTX = np.dot(trainX.T, trainX)
XTY = np.dot(trainX.T, trainY)

for i in range(epoch):
    if i != 0 and i % lossPrintGap == 0:
        print('Iteration: %7d, Loss: %10.6f' % (i, loss(w, trainX, trainY)))
    grad = 2 * (np.dot(XTX, w) - XTY + lb * w)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    mhat = m / (1 - beta1 ** (i+1))
    vhat = v / (1 - beta2 ** (i+1))
    w -= lr * mhat / (np.sqrt(vhat) + epsilon)

finalLoss = loss(w, trainX, trainY)
print('Iteration: %7d, Loss: %10.6f' % (epoch, finalLoss))
print('\nFinish training.')
print('Final Loss:', finalLoss)

np.save('model_best', w)
print('The model was written to ' + 'model_best' + '.npy')

np.save('mean_best', mean)
np.save('std_best', std)
print('The mean was written to ' + 'mean_best' + '.npy')
print('The std was written to ' + 'std_best' + '.npy')
