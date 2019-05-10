#Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling
import jieba
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

torch.manual_seed(4325)
np.random.seed(4325)

jieba.set_dictionary(sys.argv[2])
jieba.enable_parallel()

def loadX(path):
    with open(path, 'r') as f:
        content = []
        for line in f:
            content.append(','.join(line.split(',')[1:])[:-1])
    content = content[1:]
    return content

X = loadX(sys.argv[1])

t = time.time()
print('Segmentation')
#X = [' '.join(jieba.cut(x)) for x in X]
X = ' '.join(jieba.cut('\n'.join(X))).split('\n')
X = [x.strip() for x in X]
print(time.time() - t)
t = time.time()

print('Lowercasing')
X = [x.lower() for x in X]
print(time.time() - t)
t = time.time()

OX = X

print('Encoding words')
with open('1v.pkl', 'rb') as f:
    vocab_to_int = pickle.load(f)
X = [[vocab_to_int[w] for w in x.split() if w in vocab_to_int] for x in OX]
print(time.time() - t)
t = time.time()

X_lengths = [len(x) for x in X]

seq_len = 50

X = [x[:seq_len] for x in X]
X = [np.array(x) for x in X]

def padding(X):
    X_lengths = [len(x) for x in X]
    padded_X = np.zeros((len(X), seq_len))
    for i, x_len in enumerate(X_lengths):
        padded_X[i, 0:x_len] = X[i]
    return padded_X, X_lengths
X, X_lengths = padding(X)

X = torch.from_numpy(X)
test_data = TensorDataset(X)

print(X.shape[0])

batch_size = 128

test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=4)

class RNN(nn.Module):
    def __init__(self, vocab_size, seq_len, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout1 = nn.Dropout(0.5)

        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.2)

        self.conv = nn.Conv2d(1, 100, 3)
        self.maxpooling = nn.MaxPool2d(2)

        self.dropout3 = nn.Dropout(0.4)

        self.fc = nn.Linear(((seq_len - 2) // 2) * ((hidden_dim - 2) // 2) * 100, 1)

        self.sig = nn.Sigmoid()

        self.hidden = nn.Parameter(torch.randn(self.n_layers * 2, 1, self.hidden_dim))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.embedding(x)
        x = self.dropout1(x)
        # x = (batch, seq_len, input_size)
        x, h = self.gru(x, self.hidden.repeat(1, batch_size, 1))
        x = self.dropout2(x)
        # x = (batch, seq_len, 2 * hidden_size)
        # h = (num_layer * 2, batch, hidden_size)
        x = x.contiguous()
        # x = (batch, seq_len, 2 * hidden_size)
        x = x[:, :, :self.hidden_dim] + x[:, :, self.hidden_dim:]
        # x = (batch, seq_len, hidden_size)
        x = x.unsqueeze(1)
        # x = (batch, 1, seq_len, hidden_size)
        x = self.conv(x)
        # x = (batch, 1, h, w)
        x = self.maxpooling(x)
        x = self.dropout3(x)
        # x = (batch, 1, h, w)
        x = x.view(batch_size, -1)
        # x = (batch, h * w)
        x = self.fc(x)
        # x = (batch, 1)
        x = self.sig(x)
        return x

vocab_size = len(vocab_to_int)+1
embedding_dim = 300
hidden_dim = 64
n_layers = 2
net = RNN(vocab_size, seq_len, embedding_dim, hidden_dim, n_layers)
net.load_state_dict(torch.load('1.pkl'))

print(net)

X = X.type(torch.LongTensor)

if is_cuda:
    net = net.cuda()

net.eval()

t = time.time()
y_preds = np.empty((X.shape[0]))
for i, inputs in enumerate(test_loader):
    inputs = inputs[0].type(torch.LongTensor)
    if is_cuda:
        inputs = inputs.cuda()
    output = net(inputs)
    y_preds_batch = output >= 0.5
    y_preds[batch_size * i:batch_size * (i+1)] = y_preds_batch.cpu().numpy().ravel()

y_preds1 = y_preds

#---------------

torch.manual_seed(4325)
np.random.seed(4325)

t = time.time()
print('Encoding words')
with open('2v.pkl', 'rb') as f:
    vocab_to_int = pickle.load(f)
X = [[vocab_to_int[w] for w in x.split() if w in vocab_to_int] for x in OX]
print(time.time() - t)
t = time.time()

X_lengths = [len(x) for x in X]

seq_len = 70

X = [x[:seq_len] for x in X]
X = [np.array(x) for x in X]

X, X_lengths = padding(X)
X = torch.from_numpy(X)
test_data = TensorDataset(X)

print(X.shape[0])

batch_size = 128

test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=4)

class RNN(nn.Module):
    def __init__(self, vocab_size, seq_len, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout1 = nn.Dropout(0.5)

        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)

        self.fc = nn.Linear(2 * n_layers * hidden_dim, 1)

        self.sig = nn.Sigmoid()

        self.hidden = nn.Parameter(torch.randn(self.n_layers * 2, 1, self.hidden_dim))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.embedding(x)
        x = self.dropout1(x)
        # x = (batch, seq_len, input_size)
        x, h = self.gru(x, self.hidden.repeat(1, batch_size, 1))
        x = self.dropout2(x)
        # x = (batch, seq_len, 2 * hidden_size)
        # h = (num_layer * 2, batch, hidden_size)

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

vocab_size = len(vocab_to_int)+1
embedding_dim = 300
hidden_dim = 64
n_layers = 2
net = RNN(vocab_size, seq_len, embedding_dim, hidden_dim, n_layers)
net.load_state_dict(torch.load('2.pkl'))

print(net)

X = X.type(torch.LongTensor)

if is_cuda:
    net = net.cuda()

net.eval()

t = time.time()
y_preds = np.empty((X.shape[0]))
for i, inputs in enumerate(test_loader):
    inputs = inputs[0].type(torch.LongTensor)
    if is_cuda:
        inputs = inputs.cuda()
    output = net(inputs)
    y_preds_batch = output >= 0.5
    y_preds[batch_size * i:batch_size * (i+1)] = y_preds_batch.cpu().numpy().ravel()

y_preds2 = y_preds

#---------------------

torch.manual_seed(4325)
np.random.seed(4325)

codename = '3'
seq_len = 100
embedding_dim = 100
hidden_dim = 100
batch_size = 128

n_layers = 1

jieba.set_dictionary(sys.argv[2])
jieba.enable_parallel()

X = loadX(sys.argv[1])

t = time.time()
print('Segmentation')
X = ' '.join(jieba.cut('\n'.join(X))).split('\n')
X = [x.strip() for x in X]
print(time.time() - t)
t = time.time()

print('Lowercasing')
X = [x.lower() for x in X]
print(time.time() - t)
t = time.time()

print('words to indices')
with open(codename + 'v.pkl', 'rb') as f:
    word2idx = pickle.load(f)

def text2index(corpus):
    new_corpus = [[word2idx.get(word, 0) for word in doc] for doc in corpus]
    return np.array(new_corpus)

X = text2index(X)

X_lengths = [len(x) for x in X]
#X = [X[i] for i, l in enumerate(X_lengths) if l > 0]

X = [x[:seq_len] for x in X]
X = [np.array(x) for x in X]

X, X_lengths = padding(X)

test_data = TensorDataset(torch.from_numpy(X))

test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=4)

class RNN(nn.Module):
    def __init__(self, seq_len, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        #self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddingMatrix).float(), freeze=False)
        self.embedding = nn.Embedding(len(word2idx)+1, embedding_dim, padding_idx=0)

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

net = RNN(seq_len, embedding_dim, hidden_dim, n_layers)
net.load_state_dict(torch.load(codename + '.pkl'))

print(net)

if is_cuda:
    net = net.cuda()

net.eval()

t = time.time()
y_preds = np.empty((X.shape[0]))
for i, inputs in enumerate(test_loader):
    inputs = inputs[0].type(torch.LongTensor)
    if is_cuda:
        inputs = inputs.cuda()
    output = net(inputs)
    y_preds_batch = output >= 0.5
    y_preds[batch_size * i:batch_size * (i+1)] = y_preds_batch.cpu().numpy().ravel()

y_preds3 = y_preds

#---------------------

torch.manual_seed(384729834)
np.random.seed(384729834)

codename = '4'
seq_len = 256
embedding_dim = 32
hidden_dim = 32
batch_size = 128

n_layers = 1

jieba.set_dictionary(sys.argv[2])
jieba.enable_parallel()

X = loadX(sys.argv[1])

t = time.time()
print('Segmentation')
X = ' '.join(jieba.cut('\n'.join(X))).split('\n')
X = [x.strip() for x in X]
print(time.time() - t)
t = time.time()

print('Lowercasing')
X = [x.lower() for x in X]
print(time.time() - t)
t = time.time()

print('words to indices')
with open(codename + 'v.pkl', 'rb') as f:
    word2idx = pickle.load(f)

X = text2index(X)

X_lengths = [len(x) for x in X]
#X = [X[i] for i, l in enumerate(X_lengths) if l > 0]

X = [x[:seq_len] for x in X]
X = [np.array(x) for x in X]

X, X_lengths = padding(X)

test_data = TensorDataset(torch.from_numpy(X))

test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=4)

class RNN(nn.Module):
    def __init__(self, seq_len, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        #self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddingMatrix).float(), freeze=False)
        self.embedding = nn.Embedding(len(word2idx)+1, embedding_dim, padding_idx=0)

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

net = RNN(seq_len, embedding_dim, hidden_dim, n_layers)
net.load_state_dict(torch.load(codename + '.pkl'))

print(net)

if is_cuda:
    net = net.cuda()

net.eval()

t = time.time()
y_preds = np.empty((X.shape[0]))
for i, inputs in enumerate(test_loader):
    inputs = inputs[0].type(torch.LongTensor)
    if is_cuda:
        inputs = inputs.cuda()
    output = net(inputs)
    y_preds_batch = output >= 0.5
    y_preds[batch_size * i:batch_size * (i+1)] = y_preds_batch.cpu().numpy().ravel()

y_preds4 = y_preds

#----------------------

torch.manual_seed(384729834)
np.random.seed(384729834)

codename = '5'
seq_len = 256
embedding_dim = 100
hidden_dim = 64
batch_size = 128

n_layers = 2

jieba.disable_parallel()
moreWords = ['渣男', '渣女', '台女', '台男', '櫻花妹', '台灣', '異男', '雙修', '雙標', '迷妹', '低卡', '大葉', '女森', '正妹', '魯蛇', '臭鮑', '肥宅', '宅男', '宅宅', '龍妹', '子瑜', '舔共', '瞎妹']

jieba.set_dictionary(sys.argv[2])

for word in moreWords:
    jieba.add_word(word)

# must after adding words
jieba.enable_parallel()

X = loadX(sys.argv[1])

t = time.time()
print('Segmentation')
X = ' '.join(jieba.cut('\n'.join(X))).split('\n')
X = [x.strip() for x in X]
print(time.time() - t)
t = time.time()

print('Lowercasing')
X = [x.lower() for x in X]
print(time.time() - t)
t = time.time()

print('words to indices')
with open(codename + 'v.pkl', 'rb') as f:
    word2idx = pickle.load(f)

X = text2index(X)

X_lengths = [len(x) for x in X]
#X = [X[i] for i, l in enumerate(X_lengths) if l > 0]

X = [x[:seq_len] for x in X]
X = [np.array(x) for x in X]

X, X_lengths = padding(X)

test_data = TensorDataset(torch.from_numpy(X))

test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=4)

class RNN(nn.Module):
    def __init__(self, seq_len, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        #self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddingMatrix).float(), freeze=False)
        self.embedding = nn.Embedding(len(word2idx)+1, embedding_dim, padding_idx=0)

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

net = RNN(seq_len, embedding_dim, hidden_dim, n_layers)
net.load_state_dict(torch.load(codename + '.pkl'))

print(net)

if is_cuda:
    net = net.cuda()

net.eval()

t = time.time()
y_preds = np.empty((X.shape[0]))
for i, inputs in enumerate(test_loader):
    inputs = inputs[0].type(torch.LongTensor)
    if is_cuda:
        inputs = inputs.cuda()
    output = net(inputs)
    y_preds_batch = output >= 0.5
    y_preds[batch_size * i:batch_size * (i+1)] = y_preds_batch.cpu().numpy().ravel()

y_preds5 = y_preds

#----------------------

y_preds_sum = y_preds1 + y_preds2 + y_preds3 + y_preds4 + y_preds5
y_preds_ensemble = y_preds_sum >= 2.9

with open(sys.argv[3], 'w') as f:
    f.write("id,label\n")
    for i in range(y_preds.shape[0]):
        f.write('{},{}\n'.format(i, int(y_preds_ensemble[i])))
print('Done')

