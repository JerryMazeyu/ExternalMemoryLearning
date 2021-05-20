import torch
import torch.optim as optim
from torch import nn as nn
from model.simple import SimpleNet
from buildData import train_dataloader
from config import opt

net = SimpleNet()
net.to(opt.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(opt.device), labels.to(opt.device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

save_path = './ckpt/simple_net.pth'
torch.save(net.state_dict(), save_path)
print(f"Model Has Been Saved At {save_path}.")