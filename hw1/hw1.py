import sys
import numpy as np
import pandas as pd

print('Loading model...')
w = np.load('model.npy')
mean = np.load('mean.npy')
std = np.load('std.npy')

print('Loading data...')
data = pd.read_csv(sys.argv[1], index_col=[0, 1], header=None)
data = data.replace('NR', 0)
data = data.to_numpy(dtype=float)

testX = np.empty((data.shape[0] // 18, data.shape[1] * 18))
for i in range(0, data.shape[0], 18):
    testX[i // 18, :] = data[i:i+18, :].reshape(1, -1)

#testX = np.hstack([testX, testX ** 2])

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
