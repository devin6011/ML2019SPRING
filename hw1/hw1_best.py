import sys
import numpy as np
import pandas as pd

print('Loading model...')
w = np.load('model_best.npy')
mean = np.load('mean_best.npy')
std = np.load('std_best.npy')

print('Loading data...')
data = pd.read_csv(sys.argv[1], index_col=[0, 1], header=None)
data = data.replace('NR', 0)
data = data.replace('-1', 16)
data = data.to_numpy(dtype=float)

testX = np.empty((data.shape[0] // 18, data.shape[1] * 18))
for i in range(0, data.shape[0], 18):
    testX[i // 18, :] = data[i:i+18, :].reshape(1, -1)

#testX = np.hstack([testX, testX ** 2])
testX = np.hstack([testX[:, 9:135]])
temp = []
for i in [1, 3, 5, 6, 7, 8, 9, 11, 13]:
    temp.append((testX[:, i*9+8] ** 2).reshape(-1, 1))
for i in range(6, 9):
    for j in [0, 1, 7, 9]:
        temp.append((testX[:, 8 * 9 + i] * testX[:, j*9+i]).reshape(-1, 1))
temp = np.hstack(temp)
testX = np.hstack([testX, temp])

testX = (testX - mean) / std
testX = np.hstack([np.ones((testX.shape[0], 1)), testX])

print('Predicting...')
predict = np.dot(testX, w)

print('Outputting...')
with open(sys.argv[2], 'w') as outputFile:
    outputFile.write('id,value\n')
    for i in range(predict.shape[0]):
        outputFile.write('id_' + str(i) + ',' + str(predict[i, 0]) + '\n')

print('Finished...')
