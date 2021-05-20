import numpy as np
from sklearn.manifold import TSNE
from model.simple import SimpleNet
from buildData import train_dataloader, test_dataloader
import torch
from collections import defaultdict
from matplotlib import pyplot as plt


net = SimpleNet()
net.load_state_dict(torch.load('./ckpt/simple_net.pth'))

outputRes = defaultdict(list)

for ind, data in enumerate(test_dataloader):
    input, label = data
    output = net(input)
    for ii in range(len(label)):
        key = int(label[ii])
        value = output[ii,:].detach().numpy()
        outputRes[key].append(value)
    if ind >= 1:
        break


def getTSNE(resDict):
    tsne = TSNE()
    tmpy = []
    tmp = []
    for (k, v) in resDict.items():
        tmp.append(v)
        tmpy += [k for _ in range(len(v))]
    X = np.vstack(tmp)
    X_ = tsne.fit_transform(X)
    tsneDict = {}
    for i in list(resDict.keys()):
        tsneDict[i] = X_[[x for x in range(len(tmpy)) if tmpy[x] == i], :]
    return tsneDict

tsneDict = getTSNE(outputRes)

colors = ['b', 'c', 'y', 'm', 'r', 'g', 'purple', 'gold', 'black', 'orange']
for ind, (k, v) in enumerate(tsneDict.items()):
    plt.scatter(v[:,0], v[:,1], color=colors[ind], label=str(k))

plt.legend()

plt.show()