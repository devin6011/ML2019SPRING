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

torch.manual_seed(1126)
np.random.seed(1126)

codename = 'model'
batch_size = 256
latent_dim = 24

is_cuda = torch.cuda.is_available()

preprocess = transforms.Compose([
    transforms.ToTensor(),
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

print(model)
print('Parameters:', sum(p.numel() for p in model.parameters()))
print('Trainable Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

print('Loading model weights')
t = time.time()
model.load_state_dict(torch.load(codename + '.pkl'))
model.eval()
print('Time:', time.time() - t)

print('Predicting')
testData = TensorDataset(imgs)
testLoader = DataLoader(testData, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
t = time.time()
labels = []
for (inputs,) in testLoader:
    if is_cuda:
        inputs = inputs.cuda()
    cluster = model.predict(inputs)
    labels.append(cluster.detach().cpu())
labels = torch.cat(labels).numpy()
print('Time:', time.time() - t)

print('Start output')
t = time.time()
testCase = pd.read_csv(sys.argv[2], index_col=0).to_numpy() - 1
ans = labels[testCase[:, 0]] == labels[testCase[:, 1]]
ans = pd.DataFrame(ans.astype(int), columns=['label'])
ans.to_csv(sys.argv[3], index_label='id')
print('End output')
print('Time:', time.time() - t)
