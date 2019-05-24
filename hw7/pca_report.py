import os
import sys
import time
import numpy as np
from skimage.io import imread, imsave

#Number of principal components
k = 5
imgNames = [str(i) + '.jpg' for i in range(415)]
imgPath = sys.argv[1]

def postprocess(img):
    x = img.copy()
    x -= np.min(x)
    x /= np.max(x)
    x = (x * 255).astype(np.uint8)
    return x

print('Reading images')
t = time.time()
imgs = []
for imgName in imgNames:
    img = imread(os.path.join(imgPath, imgName))
    imgs.append(img)
imgShape = imgs[0].shape
print('Time:', time.time() - t)

print('Preprocessing')
t = time.time()
imgs = [img.flatten() for img in imgs]
trainData = np.array(imgs).astype(np.float32)
mean = np.mean(trainData, axis=0)
trainData -= mean
print('Time:', time.time() - t)

print('Computing SVD')
t = time.time()
U, s, VH = np.linalg.svd(trainData, full_matrices=False)
'''
input: M * N
U: M * K
S(Sigma): K
VH(H means *): K * N
To make S diagonal matrix: np.diag(S)

PCA: C = XTX/n or XTX / (n-1)
     C = W \Lambda W^{-1}
     X_k = X W_k (Project to lower dimension)
SVD: X = U S VH

     C = V (S^2 / (n-1)) V^T = V (S^2 / (n-1)) V^{-1}
     \Lambda = S^2 / (n-1)

     T = X V = U S VH V = U S
     T_k = U_k S_k = X W_k
'''
print('Time:', time.time() - t)

# Report Problem a
average = postprocess(mean)
imsave('average.jpg', average.reshape(imgShape))

# Report Problem b
for i in range(5):
    eigenface = postprocess(VH[i, :])
    imsave('{}_eigenface.jpg'.format(i), eigenface.reshape(imgShape))

# Report Problem c
testImgIDs = [78, 147, 242, 341, 353]
for testImgID in testImgIDs:
    reconstruction = postprocess(U[testImgID, :5] * s[:5] @ VH[:5, :] + mean)
    imsave('{}_reconstruction.jpg'.format(testImgID), reconstruction.reshape(imgShape))

# Report Problem d
print('Problem d')
for i in range(5):
    print('{:.1f}'.format(s[i] * 100 / sum(s)))
