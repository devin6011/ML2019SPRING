#Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling
import jieba
#import emoji
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import time
import string
import pickle
import sys
from collections import Counter

is_cuda = torch.cuda.is_available()

torch.manual_seed(4325)
np.random.seed(4325)


jieba.set_dictionary(sys.argv[4])
jieba.enable_parallel()

def loadX():
    with open(sys.argv[1], 'r') as f:
        content = []
        for line in f:
            content.append(','.join(line.split(',')[1:])[:-1])
    content = content[1:]
    return content

def loadY():
    with open(sys.argv[2], 'r') as f:
        content = []
        for line in f:
            content.append(line.split(',')[1][:-1])
    content = content[1:]
    content = list(map(int, content))
    return content

X = loadX()
Y = loadY()
#X = [emoji.demojize(x) for x in X]


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

#print('Removing punctuation')
#punctuationEng = string.punctuation
#punctuationHan = '，。：、；？！～…（）—「」『』．‧'
#punctuation = set(punctuationEng + punctuationHan)
#X = [''.join([c for c in x if c not in punctuation]) for x in X]
#print(time.time() - t)
#t = time.time()

#print('Removing digits')
#digits = set(string.digits)
#X = [''.join([c for c in x if c not in digits]) for x in X]
#print(time.time() - t)
#t = time.time()

print('Encoding words')
words = ' '.join(X).split()
count_words = Counter(words)
print(len(words))
sorted_words = count_words.most_common()
print(len(sorted_words))
vocab_to_int = {w:i+1 for i, (w, c) in enumerate(filter(lambda x : 3 <= x[1] <= 10000, sorted_words))}
print(len(vocab_to_int))
print(max(vocab_to_int.values()))
X = [[vocab_to_int[w] for w in x.split() if w in vocab_to_int] for x in X]
with open('1v.pkl', 'wb') as f:
    pickle.dump(vocab_to_int, f)
print(time.time() - t)
t = time.time()

X_lengths = [len(x) for x in X]
X = [X[i] for i, l in enumerate(X_lengths) if l > 0]
Y = [Y[i] for i, l in enumerate(X_lengths) if l > 0]

seq_len = 50

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

split_frac = 0.88
train_x = X[0:int(split_frac*len(X))]
train_y = Y[0:int(split_frac*len(X))]
valid_x = X[int(split_frac*len(X)):]
valid_y = Y[int(split_frac*len(X)):]

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))

batch_size = 10

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, num_workers=4)

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
# 75.2 when = 64
# 64, embedding=300: 75.5
# 300, embedding=300: ~74
# 64, 300, seqlen=50: 75.79
n_layers = 2
net = RNN(vocab_size, seq_len, embedding_dim, hidden_dim, n_layers)

print(net)
print('Parameters:', sum(p.numel() for p in net.parameters()))
print('Trainable Parameters:', sum(p.numel() for p in net.parameters() if p.requires_grad))

lr = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

epochs = 10

counter = 0
print_every = 1000
clip = 5

if is_cuda:
    net = net.cuda()

net.train()

bestAcc = 0.0

t = time.time()
for epoch in range(epochs):
    for inputs, labels in train_loader:
        counter += 1
        print(counter, end='\r')
        inputs = inputs.type(torch.LongTensor)
        if is_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        net.zero_grad()
        output = net(inputs)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        if counter % print_every == 0:
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
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)),
                    "Accuracy: {:.6f}".format(valAcc),
                    "Time: {}".format(time.time() - t))
            if valAcc > bestAcc:
                weightName = '1.pkl'
                print("Find Better Model, Saving the model as " + weightName)
                bestAcc = valAcc
                torch.save(net.state_dict(), weightName)
            t = time.time()
