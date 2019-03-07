import numpy as np
import pandas as pd

np.random.seed(414823)
epoch = 500000
lossPrintGap = 5000
lr = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
lb = 1.0

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
        trainX.append(x)
        trainY.append(y)

trainX = np.vstack(trainX)
trainY = np.vstack(trainY)

# feature scaling
mean = np.mean(trainX, axis=0)
std = np.std(trainX, axis=0)
trainX = (trainX - mean) / std
np.save('mean', mean)
np.save('std', std)
print('The mean was written to ' + 'mean' + '.npy')
print('The std was written to ' + 'std' + '.npy')

# add bias terms
trainX = np.hstack([np.ones((trainX.shape[0], 1)), trainX])
temp = np.hstack([trainX, trainY])
np.random.shuffle(temp)

# linear regression (Adam)
def loss(w, trainX, trainY):
    return np.sqrt(np.mean(np.square(np.dot(trainX, w) - trainY)))

print('Start training...')
print('Epoch:', epoch)
print('Initial learning rate:', lr)
print('Initial beta1:', beta1)
print('Initial beta2:', beta2)
print('Lambda:', lb)

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
np.save('model', w)
print('The model was written to ' + 'model' + '.npy')
