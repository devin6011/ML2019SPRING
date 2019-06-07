import models

import numpy as np
import pandas as pd
import io
import keras
import sys

outputFilename = sys.argv[2]
print('Loading data...')
testData = pd.read_csv(sys.argv[1], skiprows=[0], header=None)
X_test = pd.read_csv(io.StringIO(testData.iloc[:, 1].to_csv(header=False, index=False)), sep='\s+', header=None).to_numpy(dtype=float)
id_test = testData.iloc[:, 0].to_numpy(dtype=float)
print('Loading data done')

print('Preprocessing...')

X_test /= 255
X_test = X_test.reshape((-1, 48, 48, 1))

print('Preprocessing done')

print('Loading models...')

model = models.Model1()
with np.load('model.npz') as weightFile:
    weights = weightFile['arr_0']
weights = [x.astype(np.float32) for x in weights]
model.set_weights(weights)

print('Loading models done')

print('Predicting...')

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print('Predicting done')

print('Outputting...')

with open(outputFilename, 'w') as outputFile:
    outputFile.write('id,label\n')
    for i in range(y_pred.shape[0]):
        outputFile.write(str(i) + ',' + str(y_pred[i]) + '\n')

print('Outputting done')
