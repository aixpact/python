
# coding: utf-8

# In[196]:


from __future__ import print_function, division
import argparse
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
from PIL import Image
import random
import shutil
import sys
import time


import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
#from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms

# plot inline
# get_ipython().run_line_magic('matplotlib', 'inline')

# set seeds for reproduction
np.random.seed(0)
torch.manual_seed(0)

# interactive mode on
# plt.ion()


# In[197]:


# Check data directory
data_dir = 'fashionmnist'

from subprocess import check_output
print(check_output(["ls", data_dir]).decode("utf8"))
# Any results you write to the current directory are saved as output.

# TODO
import torchvision.transforms as transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
}

# Build loaders
M_BATCH = 8
WORKERS = 4
phases = ['train', 'val']

# download to processed and raw folders
image_datasets = {x: datasets.FashionMNIST(data_dir, download=True, transform=data_transforms[x]) 
                  for x in phases}

dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=M_BATCH,
                                             shuffle=True, num_workers=WORKERS)
              for x in phases}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(image_datasets['train'])

print(class_names)
print(next(iter(dataloaders['train']))[1])
print(image_datasets['train'])




# In[198]:


#
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0)) # convert to np
    plt.imshow(inp)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])


# In[199]:


# 


# In[200]:


class FashionMNIST(data.Dataset):
    """
    read fashionMNIST
    
    csv file: labels = [0, :], pixels = [1:, :]
    
    """
    def __init__(self, path, transform):
        self.transform = transform
        data = pd.read_csv(path)
        
        self.images = data.iloc[:, 1:].values
        self.image_size = self.images.shape[1]
        self.image_height = np.ceil(np.sqrt(self.image_size)).astype(np.uint8)
        self.image_width = self.image_height
        
        self.labels = data.iloc[:, 0].values.ravel()
        self.labels_classes = np.unique(self.labels)
        
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = image.reshape((self.image_width, self.image_height)).astype(np.uint8)
        image = Image.fromarray(image, mode='L')
        image = self.transform(image)
        return image, label
    
    def __len__(self):
        return self.images.shape[0]




# In[201]:


# Build loaders
# M_BATCH = 4
# WORKERS = 4
# 
# import torchvision.transforms as transforms
# transforms = transforms.Compose([transforms.ToTensor()])
# 
# trainset = FashionMNIST(
#     './fashionmnist/train/fashion-mnist_train.csv', transforms)
# 
# train_loader = DataLoader(
#     trainset, batch_size=M_BATCH,shuffle=True, num_workers=WORKERS)
# 
# valset = FashionMNIST(
#     './fashionmnist/val/fashion-mnist_test.csv', transforms)
# 
# val_loader = DataLoader(
#     valset, batch_size=M_BATCH,shuffle=False, num_workers=WORKERS)
# 
# dataloaders = {'train': train_loader, 'val': val_loader}


# In[202]:


#


# In[203]:


#


# In[204]:


class FashionMnistNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 7 * 7)
        x = self.classifier(x)
        return x


# In[227]:


#
nepochs = 4
net = FashionMnistNet()
print(net)

#
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


# In[237]:


#
def train_val(net, loader, scheduler, criterion, optimizer, phase):
    net.train(phase == 'train')
    running_loss = 0
    running_accuracy = 0

    for i, (X, y) in enumerate(loader):
        #X, y = Variable(X), Variable(y)
        # http://pytorch.org/docs/master/notes/autograd.html#volatile
        X, y = Variable(X, volatile=(phase == 'val'), requires_grad=(phase == 'train')), Variable(y)
        
        output = net(X)
        loss = criterion(output, y)
        
        if phase == 'train':
            #scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        running_loss += loss.data[0]
        pred = output.data.max(1, keepdim=True)[1]
        running_accuracy += pred.eq(y.data.view_as(pred)).cpu().sum()
        
    return running_loss/len(loader), running_accuracy/len(loader.dataset)


# In[238]:


# One run
# len(list(dataloaders['train']))  ## 7500
#train_val(net, dataloaders['train'], scheduler, criterion, optimizer, 'train')


# In[260]:


# Helper functions for printing stats
def time_format(secs):
    """Convert seconds to h:mm:ss"""
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def print_header():
    """Print header"""
    h_template = """{:8}\t\t {:8}\t\t    {:12}\t {:8}\t\t {:8}"""
    print()
    print(h_template.format('Phase', 'Epoch', 'Loss', 'Accurracy', 'Duration'))
       
       
def print_stat(phase, epoch, loss, acc, duration):
    """"""
    p_template = """{:8}\t\t {:8}\t\t {:8.4f}\t\t    {:8.1f}\t\t {:8}"""
    print(p_template.format(phase, epoch, loss, acc*100, time_format(duration)))


# In[261]:


# Train model
for epoch in range(nepochs):
    start = time.time()
    print_header()

    for phase in ['train', 'val']:
        loss, acc = train_val(net, dataloaders[phase], scheduler, criterion, optimizer, phase)
    
        if phase == 'val':
            scheduler.step(loss)

        end = time.time()
        print_stat(phase, epoch, loss, acc, end-start)

