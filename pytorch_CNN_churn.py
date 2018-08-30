
"""
Title: Customer churn
Abstract: Find an ANN model to predict customer churn with a high (>90%) churn accuracy
Conclusion: baseline accuracy is 80% (20% churns). Best accuracy found todate is: 86%
Author: Frank J. Ebbers
Date: 10/01/2018
"""



# Import (Select from extensive list)
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
import seaborn as sns
import shutil
from subprocess import check_output
import sys
import time
import unittest

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

# Hyperparameters, constants
M_BATCH = 16
EPOCHS = 100
LR = 0.010
MOMENTUM = 0.9
DECAY_STEP = 20  # epoch steps between LR decay
DECAY_LR = 0.05
WORKERS = 4
PHASES = ['train', 'val']
tune_template = 'BA{}_EP{}_LR{}_MO{}'
tuning_params = tune_template.format(M_BATCH, EPOCHS, LR, MOMENTUM).replace('.', '')

# set seeds for reproduction
np.random.seed(0)
torch.manual_seed(0)

# Import dataset and show directory
data_dir = 'churn_data'
data_file = 'Churn_Modelling.csv'
path = os.path.join(data_dir, data_file)
try:
    dataset = pd.read_csv(path)
    print('$: root/' + str(data_dir))
    print(check_output(["ls", data_dir]).decode("utf8"))
except IOError as err:
    print("IO error: {0}".format(err))
    print("Oops! Try again...")

# EDA - explore dataset to gain intuition TODO EDA

# Setup the seaborn and matplotlib figure defaults
sns.set(rc={"figure.figsize": (6, 6)})
plt.rcParams.update({'figure.autolayout': True})

# Get features and target name(s)
target = dataset.columns.tolist()[-1:]
features = dataset.columns.tolist()

# Univariate analysis

# Explore distributions, density/frequency
def frequency(dataset):
    """"""
    length = len(dataset)
    freq_list = [[ftr, round(len(dataset[ftr].unique())/length*100, 2)] for ftr in dataset.columns]
    return freq_list

frequency(dataset)


# Helper function to create index grid for subplots
def index_grid(dataset, grid_cols):
    """"""
    nfeatures = len(dataset.columns)
    grid_rows = int(np.ceil(nfeatures/grid_cols))
    grid = np.zeros([grid_rows*grid_cols])
    grid[:nfeatures] = np.arange(0, nfeatures)
    grid = grid.reshape(grid_cols, -1)
    return grid, grid_rows, grid_cols


# Show densityplots in grid
def univariate_plot(dataset, grid_cols=4, fig_size=(9, 7)):
    """"""
    grid, grid_rows, grid_cols = index_grid(dataset, grid_cols)
    f, axes = plt.subplots(grid_rows, grid_cols, figsize=fig_size)
    cpal=sns.color_palette("Blues_d")

    for i, feature in enumerate(dataset.columns, 0):
        x, y = np.where(grid == i)
        plt.title(feature)
        if dataset[feature].dtype not in ['object']:
            sns.distplot(dataset[feature], ax=axes[x[0], y[0]])
        else:
            sns.countplot(x=feature, data=dataset, ax=axes[x[0], y[0]], palette=cpal)
    return None

univariate_plot(dataset, 4)


# Show bivariate analysis in grid TODO
def bivariate_plot(dataset, target, grid_cols=4, fig_size=(9, 7)):
    """"""
    for i, feature in enumerate(dataset.columns, 1):
        plt.figure(i)
        if dataset[feature].dtype not in ['object']:
            sns.jointplot(x=feature, y=target, data=dataset, kind="reg")
        else:
            sns.factorplot(x=target, hue=feature, data=dataset, kind='count')
    return None

bivariate_plot(dataset, target[0], 4)


# Helper functions
# Generic function to split features in numerical and categorical
def features_by_type(dataset):
    """Describe stats for each input, grouped by dtype"""
    cls_type = {'cat':['object'], 'num':['int64', 'float64' ]}
    result = {'cat':None, 'num':None}
    for cls in cls_type.keys():
        cls_cols = [c for c in dataset.columns if dataset[c].dtype in cls_type[cls]]
        result[cls] = cls_cols
    return result

# Generic function to describe inputs of dataset
def summary(dataset, cat):
    """Describe stats for each input, grouped by dtype"""
    features = features_by_type(dataset)[cat]
    sumry = [dataset[c].describe() for c in features]
    sumry = pd.DataFrame.from_records(sumry)
    sumry['features'] = features
    sumry.set_index('features', inplace=True)
    return sumry

# Drop feature
def drop_feature(dataset, feature):
    """"""
    dataset.drop(feature, axis=1, inplace=True)
    return None

# Update featurelist and return updated summary
def update_summary(dataset, type):
    """"""
    features_cat = features_by_type(dataset)[type]
    return summary(dataset, type)

# Generic function for facet scatterplot
def explore_(dataset, y_feature, row_feature, col_feature, alpha=.6):
    for x_ftr in dataset.columns:
        g = sns.FacetGrid(dataset, row=row_feature, col=col_feature, hue=target[0])
        g.map(plt.scatter, x_ftr, y_feature, alpha=alpha)
        g.add_legend()
    return None


# Set features
features_cat, features_num = features_by_type(dataset)['cat'], features_by_type(dataset)['num']

# Explore categorical features #################################
# Visual inspection and removal of weak, noisy or irrelevant features
summary(dataset, 'cat')

# Explore Surname
# Surname is too unique and irrelevant
drop_feature(dataset, 'Surname')
update_summary(dataset, 'cat')

# Explore Gender against highest correlations
explore_(dataset, 'Age', 'Gender', 'Geography')
explore_(dataset, 'Balance', 'Gender', 'Geography')
explore_(dataset, 'IsActiveMember', 'Gender', 'Geography')
# Gender is hardly relevant
drop_feature(dataset, 'Gender')
update_summary(dataset, 'cat')


# Explore continuous features #################################
# Summary inspection features
print(summary(dataset, 'num'))


# Correlation matrix
# Generic function to return the min and max correlation coeffs
# Zero out the ones on the diagonal to find min, max
def min_max_corr(corr_df):
    """"""
    corr_df = corr_df.copy(deep=True)
    assert corr_df is not corr, 'sanity check immutability failed'
    np.fill_diagonal(corr_df.as_matrix(), 0)
    return corr_df.min().min(), corr_df.max().max()


corr = dataset.corr()
corr_min, corr_max = min_max_corr(corr)
print(corr_min, corr_max)

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)  # Create False matrix in corr size
mask[np.triu_indices_from(mask)] = True    # Set upper triangle to True



# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 160, as_cmap=True)

# Show corr heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=corr_min, vmax=corr_max,
            fmt='.2f', annot_kws={'size': 8}, annot=True,
            cbar_kws={"shrink": (corr_max-corr_min)},
            )

# Show weak coninuous features
dataset.iloc[:, :2].columns



# Slice away features of no information;
inputs = dataset.iloc[:, 3:13]
labels = dataset.iloc[:, 13]
assert isinstance(inputs, pd.DataFrame), 'Sanity check dataframe'
# Get features and target names
target = dataset.columns.tolist()[-1:]
features = inputs.columns.tolist()
inputs_features = ['CreditScore',
                'Geography',
                'Gender',
                'Age',
                'Tenure',
                'Balance',
                'NumOfProducts',
                'HasCrCard',
                'IsActiveMember',
                'EstimatedSalary']





# Pairplots feature vs. target wise
# Set pallete - https://seaborn.pydata.org/tutorial/color_palettes.html
sns.palplot(sns.hls_palette(8, l=.3, s=.8))
for i, feature in enumerate(features):
    sns.pairplot(dataset, x_vars=feature, y_vars=target, diag_kind="hist", markers="*",
                 plot_kws=dict(edgecolor="b", linewidth=1), diag_kws=dict(shade=True))
plt.show()

# Pairplot all features TODO too large
# sns.pairplot(dataset, hue=str(target[0]))

# Explore and tune dataset
new_dataset = dataset.copy(deep=True)
new_dataset['Age'] = pd.cut(np.array(dataset['Age']), [0, 20, 45, 65, 99],
                            labels=["1-young", "2-adults", "3-middle age", "4-eldery"])

# Sanity check mutability
assert new_dataset['Age'][0] != dataset['Age'][0], 'dataset is still mutable'
assert new_dataset is not dataset, 'dataset is still mutable'

# Facet scatterplot
g = sns.FacetGrid(new_dataset, col='Geography', hue=target[0])
#g.map(plt.scatter, 'Gender', 'Age', alpha=.7)
g.map(plt.scatter, 'Gender', 'Age', alpha=.7)
g.add_legend()

# Facet scatterplot
g = sns.FacetGrid(new_dataset, col='Age', hue=target[0])
g.map(plt.scatter, 'Tenure', 'Geography', alpha=.7)
g.map(plt.scatter, 'CreditScore', 'Geography', alpha=.7)
g.add_legend()

# Facet scatterplot
g = sns.FacetGrid(new_dataset, col='Age', hue=target[0])
g.map(plt.scatter, 'CreditScore', 'Geography', alpha=.7)
g.add_legend()

# Facet scatterplot
g = sns.FacetGrid(new_dataset, col='Age', hue=target[0])
g.map(plt.scatter, 'Tenure', 'Balance', alpha=.7)
g.add_legend()

# Facet scatterplot
g = sns.FacetGrid(new_dataset, col='Age', hue=target[0])
g.map(plt.scatter, 'EstimatedSalary', 'Balance', alpha=.7)
g.add_legend()

# Facet scatterplot
g = sns.FacetGrid(new_dataset, col='Age', hue=target[0])
g.map(plt.scatter, 'NumOfProducts', 'Balance', alpha=.7)
g.add_legend()

# Facet scatterplot
g = sns.FacetGrid(new_dataset, col='Age', hue=target[0])
g.map(plt.scatter, 'HasCrCard', 'Balance', alpha=.7)
g.add_legend()

# Facet scatterplot
g = sns.FacetGrid(new_dataset, col='Age', hue=target[0])
g.map(plt.scatter, 'CreditScore', 'Balance', alpha=.7)
g.add_legend()

# Facet scatterplot
g = sns.FacetGrid(new_dataset, col='Age', hue=target[0])
g.map(plt.scatter, 'IsActiveMember', 'Tenure', alpha=.7)
g.add_legend()

# Feature sets from scatterplots
weak_features = ['Geography', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
strong_features = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts']

# Distribution-, histogram- and rugplot
g = sns.FacetGrid(new_dataset, row='Gender', col='Age')
g.map(sns.distplot, 'Balance', rug=True)

# Zero vs. any balance churn
mask = new_dataset['Balance'] == 0
bal_dataset = new_dataset.copy(deep=True)
bal_dataset['Balance'] = pd.cut(np.array(new_dataset['Balance']), [-10e6, -0.1, 0.1, 10e6],
                                labels=['negative', 'zero', 'positive'])
g = sns.FacetGrid(new_dataset, row='Balance', col='Age')
g.map(sns.distplot, 'CreditScore', rug=True)

# Facet barplot
g = sns.FacetGrid(new_dataset, col=target[0], sharex=False)
g.map(sns.barplot, 'Age', 'Gender')
g.add_legend()

# Facet barplot
g = sns.FacetGrid(new_dataset, col=target[0], hue='Gender', sharex=False)
g.map(sns.barplot, 'Age', 'NumOfProducts')
g.add_legend()




# Check for missing data
total = inputs.isnull().sum().sort_values(ascending=False)
percent = (inputs.isnull().sum() / inputs.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data)

########################################################################################


# Encoding categorical data
# Hard copy values
# Inputs must be immutable for hashing (LabelEncoder)
if hasattr(inputs, 'values'):
    inputs = inputs.values

# Construct LabelEncoder objects for each categorical feature
# fit=set levels and transform=set to ints
le_X_1 = LabelEncoder()
inputs[:, 1] = le_X_1.fit_transform(inputs[:, 1])  # Geography
le_X_2 = LabelEncoder()
inputs[:, 2] = le_X_2.fit_transform(inputs[:, 2])  # Gender
# encode Geography to onehotvector/dummy vars
# to avoid ordinal effect on prediction [0,1,2]
onehotencoder = OneHotEncoder(categorical_features=[1])
inputs = onehotencoder.fit_transform(inputs).toarray()
inputs = inputs[:, 1:]  # remove 1 dummy var (dummy var trap) (need n_levels - 1)


# Create training and validation sets
# Split dataset in train and validation sets
inputs_train, inputs_val, labels_train, labels_val = train_test_split(
    inputs, labels, test_size=0.2, stratify=labels, random_state=0)
# print(inputs_train[:5], inputs_val[:5], labels_train[:5], labels_val[:5])

# Normalize features
sc = StandardScaler()
inputs_train = sc.fit_transform(inputs_train)  # TODO
inputs_val = sc.transform(inputs_val)

# Prepare data for nn.Module
inputs_train = torch.Tensor(inputs_train)
inputs_val = torch.Tensor(inputs_val)
labels_train = torch.Tensor(np.array(labels_train))
labels_val = torch.Tensor(np.array(labels_val))


# 1D Tensor assigning weight to each of the classes
# TODO which way around?
weight_loss = torch.Tensor([np.mean(labels), 1-np.mean(labels)])
weight_loss_r = torch.Tensor([1-np.mean(labels), np.mean(labels)])

# Build datasets and dataloaders
# Create datasets
train_set = data.TensorDataset(inputs_train, labels_train)
val_set = data.TensorDataset(inputs_val, labels_val)

# Create loaders
train_loader = DataLoader(train_set, batch_size=M_BATCH, shuffle=True, num_workers=WORKERS)
val_loader = DataLoader(val_set, batch_size=M_BATCH, shuffle=True, num_workers=WORKERS)
dataloaders = {'train': train_loader, 'val': val_loader}


# Build model architecture
class ANN(nn.Module):
    def __init__(self, n_li, n_l1, n_l2, n_l3, n_lo):
        super(ANN, self).__init__()
        self.lin_in = nn.Linear(n_li, n_l1)
        self.lin_h1 = nn.Linear(n_l1, n_l2)
        self.lin_h2 = nn.Linear(n_l2, n_l3)
        self.lin_out = nn.Linear(n_l3, n_lo)

    def forward(self, inputs):
        out = F.relu(self.lin_in(inputs))
        out = F.relu(self.lin_h1(out))
        out = F.relu(self.lin_h2(out))
        out = F.sigmoid(self.lin_out(out))
        return out


# Number of inputs (features) and outputs (labels) for model
n_inputs = inputs_train.size()[1]
n_outputs = 2

# Construct model
model = ANN(n_inputs, 11, 9, 5, n_outputs)
# Sanity check
#print(model)


# Loss, optimizer and LR-decay
# Functions
criterion = nn.CrossEntropyLoss()  # C-class cross entropy
criterion_ = nn.CrossEntropyLoss(weight=weight_loss_r)  # C-class cross entropy, set stratified weights
criterion__ = nn.BCELoss(weight=weight_loss_r) # weights only when C >=2; output layer == 2
criterion___ = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
optimizer_ = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
optimizer__ = optim.RMSprop(model.parameters(), lr=LR, alpha=0.99, eps=1e-08, weight_decay=0, momentum=MOMENTUM)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=DECAY_STEP, gamma=DECAY_LR)


# Define training, validation
# Check for existing model, load and resume
def load_model(model, optimizer, num_epochs, resume=True):
    """Load and resume from existing model.
    :return: model path"""
    model_name = os.path.join(data_dir,
                              str(model.__class__.__name__)+ '_' +
                              str(optimizer.__class__.__name__)+ '_' +
                              tuning_params + '.pk1')
    # TODO loading doesnot work, file seems toooo small
    if os.path.exists(model_name) and resume:
        model.load_state_dict(torch.load(model_name))
    return model_name

# Sanity check: path
# load_model(model, optimizer, 20)


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
    # Cast tensors to criterion interface
    # TODO BSELoss compatible Tensor
    if 'BCELoss'in str(criterion):
        inputs = inputs.float()
        labels = labels.float().view(-1, 1)
    elif 'CrossEntropyLoss':
        labels = labels.long()

    inputs = Variable(inputs, volatile=(phase == 'val'), requires_grad=(phase == 'train'))
    labels = Variable(labels)
    
    # Compute loss and predict label(max log-probability)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    _, preds = torch.max(outputs.data, 1)
    labels = labels.data
    # TODO BSELoss compatible labels
    if 'BCELoss' in str(criterion):
        labels = torch.squeeze(labels).long()
    acc = torch.sum(preds == labels)
    return loss, acc

def train(model, loader, scheduler, criterion, optimizer, phase):
    """Training, validation for each epoch. Forward, backward props and caching metric.

    :return: loss and accuracy"""
    model.train(phase == 'train')
    cache = {'cum_count': 0, 'cum_loss': 0.0, 'cum_acc': 0.0,
             'avg_loss': 0.0, 'avg_acc': 0.0}

    print(".", end="")
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


# Generic function for training and evaluation of validation set
def eval_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """Running training and validation."""
    start = time.time()
    
    # Load last best model saved
    model_name = load_model(model, optimizer, num_epochs, resume=True)
    print(model_name)
    best_model = model.state_dict()
    best_acc = 0.0
    
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
            if phase == 'val' and acc > best_acc:
                torch.save(model.state_dict(), model_name)
                best_acc, best_model = acc, model.state_dict()

            end = time.time()
            print_stat(phase, epoch, loss, acc, end-lap)
            
    finish = time.time()            
    print_model_performance(finish-start, best_acc, model_name)

    # load best model weights
    model.load_state_dict(best_model)
    return model


# Generic print helper functions
# Helper functions for printing stats
def time_format(secs):
    """Convert seconds to h:mm:ss."""
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def print_header():
    """Print header."""
    h_template = """{:8}\t {:8}\t   {:12} {:8}\t {:8}"""
    print()
    print(h_template.format('Phase', 'Epoch', 'Loss', 'Accurracy', 'Duration'))
       
       
def print_stat(phase, epoch, loss, acc, duration):
    """Print loss, accuracy and duration at each epoch/phase."""
    p_template = """{:8} {:8}\t\t {:8.4f}\t{:8.1f}\t\t {:8}"""
    print(p_template.format(phase, epoch, loss, acc*100, time_format(duration)))
    
        
def print_model_performance(duration, best_acc, best_model):
    """Print best model performance and total duration."""
    best_acc = best_acc*100
    print('Training and validation complete in: {:8}\n'
          'Best validation Accuracy: {:2.2f}%\n'
          'Learned model saved: {:16}\n'.format(
           time_format(duration), best_acc, best_model))


# Train model
# Train and evaluate validation set
fit = eval_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=EPOCHS)


def pred_val(model):
    """Predict labels for validation set"""
    preds_cum, max_cum, count = 0.0, 0.0, 0.0
    for inputs, labels in dataloaders['val']:
        v_inputs, v_labels = Variable(inputs), Variable(labels)
        outputs = model(v_inputs)
        max, preds = torch.max(outputs.data, 1)
        preds_cum += np.mean(preds)
        max_cum += torch.mean(max)
        count += 1
    print('\nvalidation set stats:\n', '- mean preds: ', preds_cum/count,
          '\n - mean max: ', max_cum/count)
    return None

pred_val(fit)


# Evaluate performance
# Predict per batch
def pred_batch(model):
    """Predict labels for one batch"""
    inputs, labels = next(iter(dataloaders['val']))
    v_inputs, v_labels = Variable(inputs), Variable(labels)
    outputs = model(v_inputs)
    # print(outputs.data) # Todo why classes do not always sum to 1?
    _, preds = torch.max(outputs.data, 1)

    # TODO BSELoss compatible labels
    if 'BCELoss' in str(criterion):
        labels = torch.squeeze(labels).long()
    elif 'CrossEntropyLoss':
        labels = labels.long()

    print('mean labels: ', np.mean(labels))
    print('mean preds : ', np.mean(preds))

    cm = confusion_matrix(labels, preds)
    df_cm = pd.DataFrame(cm, index=[i for i in ('True Neg' ,'True Pos') ],
                         columns=[i for i in ('Pred Neg', 'Pred Pos')])
    plt.figure(figsize=(6, 4))
    plt.title('Churn Matrix - Accuracy: '+ str(np.mean(preds == labels)*100) + '%')
    sns.set(font_scale=1.4)                                  # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16})   # font size
    return np.mean(preds == labels)

# Show prediction per batch in confusion heatmap
pred_batch(fit)




#
# def progressbar(value, endvalue, bar_length=50):
#     percent = float(value) / endvalue
#     arrow = '-' * int(round(percent * bar_length) - 1) + '>'
#     spaces = ' ' * (bar_length - len(arrow))
#
#     sys.stdout.write("\rProgress: [{0}] {1}%".format(
#                      arrow + spaces, int(round(percent * 100))))
#     sys.stdout.flush()
#
# for i in range(100):
#     progressbar(i, 100)
#
# import pyprind
# import sys
# import time
#
# # https://github.com/rasbt/pyprind/issues/35
# n = 50
# bar = pyprind.ProgBar(n, stream=2) # stream=[1, 2, sys.stdout]
# for i in range(n):
#     time.sleep(0.1)
#     bar.update()
