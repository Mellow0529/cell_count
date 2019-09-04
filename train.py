import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import glob
from PIL import Image
import random
import matplotlib.pyplot as plt
from load_data import Dataset, TestDataset
from model import Net

batch_size = 1
n_iter = 150
lr = 0.001
random_seed = 60
save_path = 'save_model/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(random_seed)

dataset = Dataset()
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
print('dataset: {}'.format(len(dataset)))

mse = nn.MSELoss()
bce = nn.BCELoss()
L1_loss = nn.L1Loss()

load_model_path = 'save_model/epoch150_loss_35.7894.pth.tar'

print('==> Building model...')
net = Net()

net.load_state_dict(torch.load(load_model_path))

criterion = mse
criterion.to(device)
net.to(device)

optimizer = optim.SGD(net.parameters(), lr = lr, momentum=0.9)

print('device: {}'.format(device))

net.train()
for epoch in range(n_iter + 1):
    running_loss = 0.0
    for i, data in enumerate(loader):
        img, label = data[0], data[1]
        label = label.type(torch.FloatTensor)
        img = img.type(torch.FloatTensor)
        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()
        output = net(img)
        # print('output: {}'.format(output.shape))
        # print('label: {}'.format(label.shape))
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    show_loss_period = 5
    if epoch % show_loss_period == 0:
        print('epoch [{}/{}] running loss: {}'.format(epoch, n_iter, running_loss / show_loss_period))

    save_period = 10
    if epoch % save_period == 0:
        torch.save(net.state_dict(), os.path.join(save_path, 'epoch{}_loss_{:.4f}.pth.tar'.format(epoch, running_loss / save_period)))
        print('=== Model saved ===')
