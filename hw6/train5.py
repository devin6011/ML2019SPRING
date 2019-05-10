import jieba
#import emoji
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from gensim.models.word2vec import Word2Vec
import time
import string
import pickle
import sys
from collections import Counter

is_cuda = torch.cuda.is_available()

torch.manual_seed(384729834)
np.random.seed(384729834)

codename = '5'
seq_len = 256
embedding_dim = 100
hidden_dim = 64
split_frac = 0.9
batch_size = 128

n_layers = 2
epochs = 25 * 10
#gradient clipping
clip = 5

moreWords = ['渣男', '渣女', '台女', '台男', '櫻花妹', '台灣', '異男', '雙修', '雙標', '迷妹', '低卡', '大葉', '女森', '正妹', '魯蛇', '臭鮑', '肥宅', '宅男', '宅宅', '龍妹', '子瑜', '舔共', '瞎妹']

jieba.set_dictionary(sys.argv[4])

for word in moreWords:
    jieba.add_word(word)

# must after adding words
jieba.enable_parallel()

def loadX(path):
    with open(path, 'r') as f:
        content = []
        for line in f:
            content.append(','.join(line.split(',')[1:])[:-1])
    content = content[1:]
    return content

def loadY(path):
    with open(path, 'r') as f:
        content = []
        for line in f:
            content.append(line.split(',')[1][:-1])
    content = content[1:]
    content = list(map(int, content))
    return content

X = loadX(sys.argv[1])
testX = loadX(sys.argv[3])
Y = loadY(sys.argv[2])
#X = [emoji.demojize(x) for x in X]

# 119018-119999 are the same as 0-981
X = X[:119018]
Y = Y[:119018]

t = time.time()
print('Segmentation')
X = ' '.join(jieba.cut('\n'.join(X))).split('\n')
X = [x.strip() for x in X]
testX = ' '.join(jieba.cut('\n'.join(testX))).split('\n')
testX = [x.strip() for x in testX]
print(time.time() - t)
t = time.time()

print('Lowercasing')
X = [x.lower() for x in X]
testX = [x.lower() for x in testX]
print(time.time() - t)
t = time.time()

print('Word2Vec')
w2vModel = Word2Vec([x.split() for x in X] + [x.split() for x in testX], size=embedding_dim, iter=15, sg=1, window=10, min_count=5, workers=24)
#w2vModel.save(codename + '_w2v.model')
#print('Word2Vec model saved as ' + codename + ' _w2v.model')
#load : w2vModel = Word2Vec.load(codename + '_w2v.model')
print(w2vModel.wv.most_similar(positive=['男']))
embeddingMatrix = np.zeros((len(w2vModel.wv.vocab.items()) + 1, w2vModel.vector_size))
print('Embedding shape:', embeddingMatrix.shape)
word2idx = {}
vocab_list = [(word, w2vModel.wv[word]) for word, _ in w2vModel.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embeddingMatrix[i + 1] = vec
    word2idx[word] = i + 1

with open(codename + 'v.pkl', 'wb') as f:
    pickle.dump(word2idx, f)
print(time.time() - t)
t = time.time()

def text2index(corpus):
    new_corpus = [[word2idx.get(word, 0) for word in doc] for doc in corpus]
    return np.array(new_corpus)

X = text2index(X)

X_lengths = [len(x) for x in X]
X = [X[i] for i, l in enumerate(X_lengths) if l > 0]
Y = [Y[i] for i, l in enumerate(X_lengths) if l > 0]

X = [x[:seq_len] for x in X]
X = [np.array(x) for x in X]
Y = np.array(Y)

def padding(X):
    X_lengths = [len(x) for x in X]
    padded_X = np.zeros((len(X), seq_len))
    for i, x_len in enumerate(X_lengths):
        padded_X[i, 0:x_len] = X[i]
    return padded_X, X_lengths
X, X_lengths = padding(X)

randomIndices = np.random.shuffle(np.arange(X.shape[0]))
print(X.shape)
X = X[randomIndices][0]
Y = Y[randomIndices][0]
train_x = X[0:int(split_frac*len(X))]
train_y = Y[0:int(split_frac*len(X))]
valid_x = X[int(split_frac*len(X)):]
valid_y = Y[int(split_frac*len(X)):]

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, num_workers=4)

class RNN(nn.Module):
    def __init__(self, seq_len, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddingMatrix).float(), freeze=False)
        self.embedding.padding_idx = 0

        self.dropout1 = nn.Dropout(0.5)
        #self.layerNorm = nn.LayerNorm([seq_len, embedding_dim])

        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)

        self.fc = nn.Linear(2 * n_layers * hidden_dim, 1)
 
        self.sig = nn.Sigmoid()

        self.hidden = nn.Parameter(torch.randn(self.n_layers * 2, 1, self.hidden_dim))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.embedding(x)
        x = self.dropout1(x)
        # x = (batch, seq_len, embedding_dim)
        #x = self.layerNorm(x)
        x, h = self.gru(x, self.hidden.repeat(1, batch_size, 1))
        x = self.dropout2(x)
        # x = (batch, seq_len, 2 * hidden_size)
        # h = (num_layer * 2, batch, hidden_size)

        # h = (num_layer * 2, batch, hidden_size)
        h = torch.transpose(h, 0, 1)
        h = h.contiguous()
        # h = (batch, num_layer * 2, hidden_size)
        h = h.view(batch_size, -1)
        # h = (batch, num_layer * 2 * hidden_size)
        h = self.fc(h)
        # x = (batch, 1)
        h = self.sig(h)
        return h

#net = RNN(vocab_size, seq_len, embedding_dim, hidden_dim, n_layers)
net = RNN(seq_len, embedding_dim, hidden_dim, n_layers)

print(net)
print('Parameters:', sum(p.numel() for p in net.parameters()))
print('Trainable Parameters:', sum(p.numel() for p in net.parameters() if p.requires_grad))

criterion = nn.BCELoss()
#optimizer = optim.RMSprop(net.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer = optim.SGD(net.parameters(), lr=0.3, momentum=0.9, weight_decay=1e-5)
#optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * 25)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


counter = 0
#print_every = 1000

if is_cuda:
    net = net.cuda()

net.train()

bestAcc = 0.0

t = time.time()
for epoch in range(epochs):
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        counter += 1
        inputs = inputs.type(torch.LongTensor)
        if is_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        net.zero_grad()
        output = net(inputs)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()

        total += labels.size(0)
        y_pred = output.squeeze() >= 0.5
        correct += (y_pred == labels.byte()).sum().item()
        #print(counter, loss.item(), end='\r')
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
        scheduler.step()

        #if counter % print_every == 0:
    trainAcc = 100 * correct / total

    val_losses = []
    net.eval()
    correct = 0
    total = 0
    for inputs, labels in valid_loader:
        inputs = inputs.type(torch.LongTensor)
        if is_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        output = net(inputs)
        val_loss = criterion(output.squeeze(), labels.float())
        val_losses.append(val_loss.item())
        total += labels.size(0)
        y_pred = output.squeeze() >= 0.5
        correct += (y_pred == labels.byte()).sum().item()
    net.train()
    valAcc = 100 * correct / total
    print("Epoch: {}/{}...".format(epoch+1, epochs),
            "Step: {}...".format(counter),
            "Train Loss: {:.6f}...".format(loss.item()),
            "Train Accuracy: {:.6f}".format(trainAcc),
            "Val Loss: {:.6f}".format(np.mean(val_losses)),
            "Val Accuracy: {:.6f}".format(valAcc),
            "Time: {}".format(time.time() - t))
    #scheduler.step(np.mean(val_losses))
    if valAcc > bestAcc:
        weightName = codename + '.pkl'
        print("Find Better Model, Saving the model as " + weightName)
        bestAcc = valAcc
        torch.save(net.state_dict(), weightName)
    t = time.time()
    #scheduler.step()

    if (epoch + 1) % 25 == 0:
        optimizer = optim.SGD(net.parameters(), lr=0.3, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * 25)
