
# coding: utf-8

# ###Import libraries, modules
# 
# Sequence; core/generic, specific, modules [a-z]

# In[350]:


# Select from extensive list of imports
from __future__ import print_function, division
import argparse
import inspect
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
import torchvision.models as models
import torchvision.transforms as transforms

# plot inline
#get_ipython().run_line_magic('matplotlib', 'inline')

# set seeds for reproduction
np.random.seed(0)
torch.manual_seed(0)

# interactive mode on
#plt.ion()


# ###Define data transforms and augmentation before loading
# 
# Compose augmentation(random), transforms for training and validation/test sets.
# Set size equal to model 
# Create tensors

# In[351]:


# All torchvision pre-trained models expect input images normalized in the same way, 
# i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), 
# where H and W are expected to be at least 224. 
# The images have to be loaded into a range of [0, 1] and then normalized using 
MEAN, SD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# Generic function to build transforms
def transform_composer(img_size=0, val_size=0, **kwargs):
    """Build composed data transforms.
    
    :args: e.g.: img_size=224, val_size=256
           kwargs: boolean list of transforms in correct order:
           resize=True, c_crop=True, r_crop=True, flip=True, rotate=True, 
           tensor=True, normalize=True
    :return: composed transform"""
    train_dict = {
         'r_crop': transforms.RandomResizedCrop(img_size),
         'flip': transforms.RandomHorizontalFlip(),
         'rotate': transforms.RandomRotation(5),
         'tensor': transforms.ToTensor(),      
         'normalize': transforms.Normalize(MEAN, SD)
    }
    val_dict = {
         'resize': transforms.Resize(val_size),
         'c_crop': transforms.CenterCrop(img_size),
         'tensor': transforms.ToTensor(),      
         'normalize': transforms.Normalize(MEAN, SD)
    }
    train_transforms = [train_dict[k] for k, v in kwargs.items() if v and k in train_dict]
    val_transforms = [val_dict[k] for k, v in kwargs.items() if v and k in val_dict]
    
    data_transforms = {
        'train': transforms.Compose(train_transforms),
        'val': transforms.Compose(val_transforms)
        }
    return data_transforms  
    
    
# Compose transforms
# img_size, val_size = 224, 256
cnn_transforms = transform_composer(img_size=224, val_size=256, resize=True, c_crop=True, 
                                    r_crop=True, flip=True, rotate=True, tensor=True, normalize=True)
# Sanity check: list of transforms
print(repr(cnn_transforms['train'].__dict__))
print(repr(cnn_transforms['val'].__dict__))

# FashionMNIST
fmnist_transforms = transform_composer(flip=True, rotate=True, tensor=True)


# In[352]:


# Sanity check data directory for subfolders and files
data_dir = 'hymenoptera_data'

from subprocess import check_output
print(check_output(["ls", data_dir]).decode("utf8"))


# In[353]:


# Build loaders
M_BATCH = 4
WORKERS = 4
PHASES = ['train', 'val']
TRANSFORMS = cnn_transforms
    
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          TRANSFORMS[x])
                                          for x in PHASES}

dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=M_BATCH,
                                  shuffle=True, num_workers=WORKERS)
                                  for x in PHASES}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Sanity check images, class labels
print((repr(image_datasets['train'].__dict__))[:500])
print(dataset_sizes)
print(class_names)


# ###Visualize data

# In[354]:


# Image viewer function
def imshow(inp, title=None):
    """Show images of Tensors in Dataloader."""
    inp = inp.numpy().transpose((1, 2, 0)) # convert to np
    inp = (SD * inp) + MEAN
    inp = np.clip(inp, 0, 1)
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


# ###Build model 
# or 
# Load pretrained model

# In[355]:


# Define Network Architecture Class
# IF NOT PRELOADED


# ###Pretrained model

# In[356]:


# List Pytorch pretrained models
model_names = sorted(name for name in models.__dict__
                     if name.islower() 
                     and not name.startswith("__")
                     and callable(models.__dict__[name]))
print(model_names)


# In[360]:


# Generic function to set/define pretrained model
def pre_model(model, pretrained=True, freeze=True):
    """"""
    model = models.__dict__[model](pretrained=pretrained)
    # freeze parameters in backprop
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features    # no. of features in fc layer
    model.fc = nn.Linear(num_ftrs, 2)  # change out_features to 2 (binary loss)
    return model


# Define/set model
model_all = pre_model('resnet18', pretrained=True, freeze=False)
model_freeze = pre_model('resnet18', pretrained=True, freeze=True)

# Sanity check: show model and architecture change
print(model_all)
print(models.resnet18(pretrained=True).fc)
print(model_all.fc)


# ###Define Loss and optimizer

# In[360]:


# Loss and optimizer functions

# Hyperparameters
LR = 0.001
MOMENTUM = 0.9
DECAY_STEP = 7  # epoch steps between LR decay
DECAY_LR = 0.1

# Loss function for binary classification
criterion = nn.CrossEntropyLoss()

# Optimizer functions: optimize ALL parameters vs. final layer only
optimizer_all = optim.SGD(model_all.parameters(), lr=LR, momentum=MOMENTUM)
optimizer_freeze = optim.SGD(model_freeze.fc.parameters(), lr=LR, momentum=MOMENTUM)

# Decay LR
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_all, step_size=DECAY_STEP, gamma=DECAY_LR)


# ###Define training, validation

# In[361]:


# Check for existing model, load and resume
def load_model(model, optimizer, num_epochs, resume=True):
    """Load and resume from existing model.
    :return: model path"""
    model_name = os.path.join(data_dir,
                              str(model.__class__.__name__)+'_'+
                              str(optimizer.__class__.__name__)+'_'+
                              str(num_epochs)+'.pk1')
    if os.path.exists(model_name) and resume:
        model.load_state_dict(torch.load(model_name))
    return model_name

# Sanity check: path
load_model(model_all, optimizer_all, 20)


# In[362]:


# Generic train helper functions


def b_ward(loss, optimizer, scheduler):
    """Backpropagate loss."""
    optimizer.zero_grad()   # reset gradients
    loss.backward()         # backprop loss
    optimizer.step()        # apply gradients
    

def f_ward(model, phase, criterion, inputs, labels):
    """Forward pass.
    
    http://pytorch.org/docs/master/notes/autograd.html#volatile
    :return: loss and accuracy
    """
    inputs = Variable(inputs, volatile=(phase == 'val'), requires_grad=(phase == 'train'))
    labels = Variable(labels)
    
    # Compute loss and predict label(max log-probability)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    _, preds = torch.max(outputs.data, 1)
    acc = torch.sum(preds == labels.data)
    return loss, acc

    
def train(model, loader, scheduler, criterion, optimizer, phase):
    """Training, validation for each epoch. Forward, backward props and caching metrics.
    
    :return: loss and accuracy"""
    model.train(phase == 'train')
    cache = {'cum_count': 0, 'cum_loss': 0.0, 'cum_acc': 0.0, 
             'avg_loss': 0.0, 'avg_acc': 0.0}

    for i, (inputs, labels) in enumerate(loader):
        
        # forward
        loss, acc = f_ward(model, phase, criterion, inputs, labels)
        
        # backward
        if phase == 'train':
            b_ward(loss, optimizer, scheduler)
            
        # stats
        cache['cum_count'] += inputs.size()[0]
        cache['cum_loss'] += loss.data[0]
        cache['cum_acc'] += acc
        cache['avg_loss'] = cache['cum_loss']/cache['cum_count']
        cache['avg_acc'] = cache['cum_acc']/cache['cum_count']
    return cache['avg_loss'], cache['avg_acc']


# In[367]:


# Generic function for training and evaluation of validation set
def eval_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """Running training and validation."""
    start = time.time()
    
    # Load last best model saved
    model_name = load_model(model, optimizer, num_epochs, resume=True)
    print(model_name)
    best_model = {'model': model_name, 'best_acc': 0.0, 'best_model_wts': model.state_dict()}
    
    for epoch in range(num_epochs):
        #lap = time.time()
        print_header()
    
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            lap = time.time()
            loss, acc = train(model, dataloaders[phase], scheduler, criterion, optimizer, phase)
            
            # update LR decay
            if phase == 'val':
                scheduler.step(loss)
            # update and save best_model
            if phase == 'val' and acc > best_model['best_acc']:
                best_model['best_acc'], best_model['best_model_wts'] = acc, model.state_dict()
                torch.save(model.state_dict(), best_model['model'])
                
            end = time.time()
            print_stat(phase, epoch, loss, acc, end-lap)
            
    finish = time.time()            
    print_model_performance(finish-start, best_model)

    # load best model weights
    model.load_state_dict(best_model['best_model_wts'])
    return model


# In[368]:


# Generic print helper functions


# Helper functions for printing stats
def time_format(secs):
    """Convert seconds to h:mm:ss."""
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def print_header():
    """Print header."""
    h_template = """{:8}\t\t {:8}\t\t    {:12}\t {:8}\t\t {:8}"""
    print()
    print(h_template.format('Phase', 'Epoch', 'Loss', 'Accurracy', 'Duration'))
       
       
def print_stat(phase, epoch, loss, acc, duration):
    """Print loss, accuracy and duration at each epoch/phase."""
    p_template = """{:8}\t\t {:8}\t\t {:8.4f}\t\t    {:8.1f}\t\t {:8}"""
    print(p_template.format(phase, epoch, loss, acc*100, time_format(duration)))
    
        
def print_model_performance(duration, best_model):
    """Print best model performance and total duration."""
    print('Training and validation complete in: {:8}\n'
          'Best validation Accuracy: {:2.1f}%\n'
          'Learned model saved: {:16}\n'.format(
           time_format(duration), round(best_model['best_acc']*100), 2), best_model['model'])


# ###Train model
# 

# In[369]:


EPOCHS = 2

# Train and evaluate validation set
model_all = eval_model(model_all, criterion, optimizer_all, 
                       exp_lr_scheduler, num_epochs=EPOCHS)


# ###Evaluate performance
# 
# Tune hypermparameters

# In[370]:


# TODO


# ###Transfer learning
# 
# Only learn final layer

# ###Train weights in final layer

# In[371]:


EPOCHS = 2

model_freeze = eval_model(model_freeze, criterion, optimizer_freeze, 
                          exp_lr_scheduler, num_epochs=EPOCHS)


# ###Visualize

# ###Predict

# In[372]:


# Predict per batch
def pred_batch(model):
    """Predict labels for one batch"""
    inputs, labels = next(iter(dataloaders['val']))
    v_inputs, v_labels = Variable(inputs), Variable(labels)
    
    outputs = model(v_inputs)
    _, preds = torch.max(outputs.data, 1)
    
    return zip(inputs, preds, labels)


# Visualize predictions
def show_pred_batch(model, n_batches, n_columns=M_BATCH):
    """Show from n batches n predictions"""
    for _ in range(n_batches):
        it_batch = list(pred_batch(model))
        title = [(class_names[yhat], (yhat == y)) 
                  for _, yhat, y in it_batch][:n_columns]
        inputs = ([input for input, _, _ in it_batch])[:n_columns]
    
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs, padding=10)
        imshow(out, title)
    return None


# Show predictions
show_pred_batch(model_freeze, 5, 2)



# In[277]:


plt.ioff()

