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

model1 = models.Model1()
model2 = models.Model2()
model3 = models.Model3()
model4 = models.Model4()
model5 = models.Model5()

model1.load_weights('model1.h5')
model2.load_weights('model2.h5')
model3.load_weights('model3.h5')
model4.load_weights('model4.h5')
model5.load_weights('model5.h5')

print('Loading models done')

print('Predicting...')

y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)
y_pred5 = model5.predict(X_test)

y_pred = (y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5) / 5.0
y_pred = np.argmax(y_pred, axis=1)

print('Predicting done')

print('Outputting...')

with open(outputFilename, 'w') as outputFile:
    outputFile.write('id,label\n')
    for i in range(y_pred.shape[0]):
        outputFile.write(str(i) + ',' + str(y_pred[i]) + '\n')

print('Outputting done')
