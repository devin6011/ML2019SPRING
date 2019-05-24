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

testImageName = sys.argv[2]
testImage = imread(os.path.join(imgPath, testImageName))
testImage = testImage.flatten().astype(np.float32)
testImage -= mean

reconstruction = postprocess(testImage @ VH.T[:, :k] @ VH[:k, :] + mean)
imsave(sys.argv[3], reconstruction.reshape(imgShape))
