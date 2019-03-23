from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import pickle

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

X = loadX('./X_train', 'training')
y = loadY('./Y_train').ravel()

clf = GradientBoostingClassifier(n_estimators=300, random_state=880301, max_depth=6, learning_rate=0.05)

#scores = cross_val_score(clf, X, y, cv=4)
#print(scores.mean())

print('Training')
clf.fit(X, y)

print('Ein:', clf.score(X, y))

print('Saving model')
with open('modelE', 'wb') as f:
    pickle.dump(clf, f)

print('Done')
