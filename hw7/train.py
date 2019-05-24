import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import sys
import os
import numpy as np
import pandas as pd
import time
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

torch.manual_seed(1126)
np.random.seed(1126)

codename = 'model'
batch_size = 256
epochs = 200
maxIter = 20000
gamma = 0.1
updateInterval = 140
tolerance = 0.001
latent_dim = 24

is_cuda = torch.cuda.is_available()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    ])

postprocess = transforms.Compose([
    transforms.ToPILImage()
    ])

img_names = [str(i).zfill(6) + '.jpg' for i in range(1, 40001)]

def readImages():
    imgs = []
    for img_name in img_names:
        print('Reading', img_name, end='\r')
        img = Image.open(os.path.join(sys.argv[1], img_name))
        imgs.append(preprocess(img))
    return torch.stack(imgs)

t = time.time()
print('Reading images')
imgs = readImages()
print('Finished')
print('Time:', time.time() - t)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = nn.Sequential(
                nn.Conv2d(3, 20, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(20, 40, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(40, 80, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                )
        self.encoder2 = nn.Sequential(
                nn.Linear(80 * 4 * 4, latent_dim),
                )
        self.decoder2 = nn.Sequential(
                nn.Linear(latent_dim, 80 * 4 * 4),
                nn.ReLU(),
                )
        self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(80, 40, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(40, 20, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(20, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
                )

    def forward(self, x):
        x = self.encoder1(x)
        preFlattenSize = x.size()
        x = x.contiguous().view(x.size(0), -1)
        x = self.encoder2(x)
        
        code = x

        x = self.decoder2(x)
        x = x.contiguous().view(preFlattenSize)
        x = self.decoder1(x)
        return x, code

    def encode(self, x):
        x = self.encoder1(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.encoder2(x)
        return x

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, code_dim, alpha=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.code_dim = code_dim
        self.clusters = nn.Parameter(torch.Tensor(n_clusters, code_dim))

    def forward(self, x):
        q = 1.0 / (1.0 + torch.sum((x.unsqueeze(1) - self.clusters) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        #q = torch.t(torch.t(q) / torch.sum(q, dim=1))
        #q = q / torch.sum(q, dim=1).unsqueeze(1)
        q = q / q.sum(dim=1, keepdim=True)
        return q

class DCEC(nn.Module):
    def __init__(self, n_clusters):
        super().__init__()

        self.n_clusters = n_clusters
        self.autoencoder = Autoencoder()
        self.clusteringLayer = ClusteringLayer(n_clusters=self.n_clusters, code_dim=latent_dim)

    def forward(self, x):
        reconstruction, code = self.autoencoder(x)
        cluster = self.clusteringLayer(code)
        return (reconstruction, code, cluster)

    def predict(self, x):
        code = self.autoencoder.encode(x)
        cluster = self.clusteringLayer(code)
        return cluster.argmax(dim=1)

    @staticmethod
    def target_distribution(q):
        p = q ** 2 / q.sum(dim=0)
        p = p / p.sum(dim=1, keepdim=True)
        return p

model = DCEC(n_clusters=2)
if is_cuda:
    model.cuda()

net = model.autoencoder
print(net)
print('Parameters:', sum(p.numel() for p in net.parameters()))
print('Trainable Parameters:', sum(p.numel() for p in net.parameters() if p.requires_grad))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())

trainData = TensorDataset(imgs)
trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
net.train()
bestLoss = np.inf
for epoch in range(epochs):
    t = time.time()

    meanLoss = 0.0
    for (inputs,) in trainLoader:
        if is_cuda:
            inputs = inputs.cuda()
        net.zero_grad()
        outputs, code = net(inputs)
        loss = criterion(outputs, inputs)
        meanLoss += loss.item()
        loss.backward()
        optimizer.step()
    meanLoss = meanLoss * batch_size / 40000
    print('Epoch: {}/{}'.format(epoch+1, epochs),
            'TrainLoss: {:.6f}'.format(meanLoss),
            'Time: {}'.format(time.time() - t))
    if meanLoss < bestLoss:
        bestLoss = meanLoss
        print('Find Better Model, Saving it as', codename + '.pkl')
        torch.save(net.state_dict(), codename + '.pkl')

net.load_state_dict(torch.load(codename + '.pkl'))

testLoader = DataLoader(trainData, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
net.eval()

meanLoss = 0
codes = []
for (inputs,) in testLoader:
    if is_cuda:
        inputs = inputs.cuda()
    outputs, code = net(inputs)
    loss = criterion(outputs, inputs)
    meanLoss += loss.item()
    codes.append(code.detach().cpu())
codes = torch.cat(codes).numpy()
codes_mean, codes_std = np.mean(codes, axis=0), np.std(codes, axis=0)
codes_std[codes_std == 0] = 1
codes = (codes - codes_mean) / codes_std

codes_embedded = TSNE(n_components=2, verbose=2).fit_transform(codes[:400])

plt.scatter(codes_embedded[:, 0], codes_embedded[:, 1])
plt.savefig('outputFig.jpg')
print('Loss AE:', meanLoss * batch_size / 40000)

print('Start k means')
t = time.time()
kmeans = KMeans(n_clusters=2, n_init=100, max_iter=900, random_state=1126)
y_pred = torch.tensor(kmeans.fit_predict(codes)).cuda().long()
y_pred_last = y_pred
labels = kmeans.labels_
model.state_dict()['clusteringLayer.clusters'].copy_(torch.tensor(kmeans.cluster_centers_))
print('End k means')
print('Time:', time.time() - t)

print('Start output_ae')
testCase = pd.read_csv('test_case.csv', index_col=0).to_numpy() - 1
ans = labels[testCase[:, 0]] == labels[testCase[:, 1]]
ans = pd.DataFrame(ans.astype(int), columns=['label'])
ans.to_csv('output_ae.csv', index_label='id')
print('End output_ae')
print('Time:', time.time() - t)

trainLoader2 = DataLoader(trainData, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
optimizer2 = torch.optim.Adam(model.parameters())
model.train()
trainLoaderIter = iter(trainLoader2)
saveInterval = len(trainLoader) * 5
index = 0
t = time.time()
meanLoss = 0.0
for it in range(maxIter):
    if it % updateInterval == 0:
        q = []
        for (inputs,) in testLoader:
            if is_cuda:
                inputs = inputs.cuda()
            _, _, cluster = model(inputs)
            q.append(cluster.detach())
        q = torch.cat(q)
        p = model.target_distribution(q)
        y_pred = q.argmax(dim=1)

        y_pred_diff = torch.sum(y_pred != y_pred_last).float() / y_pred.size(0)
        y_pred_last = y_pred
        if it > 0 and y_pred_diff < tolerance:
            print('y_pred_diff: {} < tolerance: {}'.format(y_pred_diff.item(), tolerance))
            print('Early stop.')
            break
    inputs = next(trainLoaderIter, None)
    if not inputs:
        trainLoaderIter = iter(trainLoader2)
        inputs = next(trainLoaderIter, None)
        index = 0
    inputs = inputs[0]
    if is_cuda:
        inputs = inputs.cuda()
    model.zero_grad()
    outputs, code, q = model(inputs)
    loss = criterion(outputs, inputs) + gamma * (p[index*batch_size:(index+1)*batch_size] * (p[index*batch_size:(index+1)*batch_size] / q).log()).sum()
    meanLoss += loss.item()
    index += 1
    loss.backward()
    optimizer2.step()
    if it % saveInterval == 0:
        meanLoss = meanLoss / saveInterval
        print('Model saved as', codename + '.pkl')
        torch.save(model.state_dict(), codename + '.pkl')
        print('Iter: {}/{}'.format(it+1, maxIter),
                'TrainLoss: {:.6f}'.format(meanLoss),
                'Time: {}'.format(time.time() - t))
        meanLoss = 0.0
        t = time.time()

print('Final model saved as', codename + '.pkl')
torch.save(model.state_dict(), codename + '.pkl')
model.load_state_dict(torch.load(codename + '.pkl'))
model.eval()

testImgs = [1, 4, 5, 6, 7, 8, 9, 10, 32, 53]
for imgId in testImgs:
    img = imgs[imgId]
    if is_cuda:
        img = img.cuda()
    img = img.unsqueeze(0)
    outputs, _, _ = model(img)
    outputs = postprocess(outputs.cpu()[0])
    outputs.save('output' + str(imgId) + '.jpg')

meanLoss = 0
codes = []
labels = []
for (inputs,) in testLoader:
    if is_cuda:
        inputs = inputs.cuda()
    outputs, code, cluster = model(inputs)
    loss = criterion(outputs, inputs)
    meanLoss += loss.item()
    codes.append(code.detach().cpu())
    labels.append(cluster.argmax(dim=1).detach().cpu())
codes = torch.cat(codes).numpy()
labels = torch.cat(labels).numpy()
codes_mean, codes_std = np.mean(codes, axis=0), np.std(codes, axis=0)
codes_std[codes_std == 0] = 1
codes = (codes - codes_mean) / codes_std

codes_embedded = TSNE(n_components=2, verbose=2).fit_transform(codes[:400])

plt.figure()
plt.scatter(codes_embedded[np.where(labels[:400] == 0)[0], 0], codes_embedded[np.where(labels[:400] == 0)[0], 1])
plt.scatter(codes_embedded[np.where(labels[:400] == 1)[0], 0], codes_embedded[np.where(labels[:400] == 1)[0], 1])
plt.savefig('outputFig2.jpg')
print('Loss DCEC:', meanLoss * batch_size / 40000)

print('Start output')
testCase = pd.read_csv('test_case.csv', index_col=0).to_numpy() - 1
ans = labels[testCase[:, 0]] == labels[testCase[:, 1]]
ans = pd.DataFrame(ans.astype(int), columns=['label'])
ans.to_csv('output.csv', index_label='id')
print('End output')
print('Time:', time.time() - t)
