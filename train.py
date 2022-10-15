import utils
import backbones

import glob
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms

m = backbones.get_model('resnet101')
m = m.to('cuda' if torch.cuda.is_available() else 'cpu')

from sklearn.metrics import f1_score, accuracy_score
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, loss_fn, batchSize, trainset, valset, optimizer, num_epochs):
  
    # Shuffling is needed in case dataset is not shuffled by default.
    train_loader = torch.utils.data.DataLoader(dataset = trainset,
                                                batch_size = batchSize,
                                                shuffle = True)
    # We don't need to bach the validation set but let's do it anyway.
    val_loader = torch.utils.data.DataLoader(dataset = valset,
                                            batch_size = batchSize,
                                            shuffle = False) # No need.

    # Define number of epochs.
    N = num_epochs

    # log accuracies and losses.
    train_accuracies = []; val_accuracies = []
    train_losses = []; val_losses = []

    # GPU enabling.
    model = model.cuda()
    loss_fn = loss_fn.cuda()


    # Training loop. Please make sure you understand every single line of code below.
    # Go back to some of the previous steps in this lab if necessary.
    all_cum_loss = 1e5
    for epoch in range(0, N):
        correct = 0.0
        cum_loss = 0.0

        # Make a pass over the training data.
        model.train()
        for (i, (name, inputs, labels)) in enumerate(train_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Forward pass. (Prediction stage)
            scores = model(inputs)
            loss = loss_fn(scores, labels)

            # Count how many correct in this batch.
            max_labels = scores > 0
            correct += sum([f1_score(a, b) for a, b in zip(max_labels.detach().cpu(), labels.detach().cpu())])
            cum_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Train-epoch %d. Iteration %05d / %05d, Avg-Loss: %.4f, Accuracy: %.4f' % 
                    (epoch, i + 1, len(train_loader), cum_loss / (i + 1), correct / ((i + 1) * batchSize)))

        train_accuracies.append(correct / len(trainset))
        train_losses.append(cum_loss / (i + 1))   

        correct = 0.0
        cum_loss = 0.0
        model.eval()
        for (i, (name, inputs, labels)) in enumerate(val_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores = model(inputs)
            cum_loss += loss_fn(scores, labels).item()

            max_labels = scores > 0
            correct += sum([f1_score(a, b) for a, b in zip(max_labels.detach().cpu(), labels.detach().cpu())])

        val_accuracies.append(correct / len(valset))
        val_losses.append(cum_loss / (i + 1))

        # Logging the current results on validation.
        print('Validation-epoch %d. Avg-Loss: %.4f, Accuracy: %.4f' % 
            (epoch, cum_loss / (i + 1), correct / len(valset)))

        if cum_loss / (i + 1) < all_cum_loss:
            torch.save(m, f'resnet_cs701_{epoch}.pt')
            all_cum_loss = cum_loss / (i + 1)


    

# Pytorch DataLoader for iterating over batches.
batchSize = 8

# Create the model.
loss_fn = nn.BCEWithLogitsLoss()

# Define a learning rate. 
learningRate = 1e-3

# Optimizer.
optimizer = torch.optim.Adam(m.parameters(), lr = learningRate)
train_model(m, loss_fn, batchSize, utils.trainset, utils.valset, optimizer, 50)

torch.save(m, 'csra_resnet50.pt')

"""#Eval"""

def eval_model(model, loss_fn, batchSize, valset):

    # We don't need to bach the validation set but let's do it anyway.
    val_loader = torch.utils.data.DataLoader(dataset = valset,
                                                batch_size = batchSize,
                                                shuffle = False) # No need.

    val_accuracies = []
    val_losses = []

    # GPU enabling.
    model = model.cuda()
    loss_fn = loss_fn.cuda()

    # Make a pass over the validation data.
    correct = 0.0
    cum_loss = 0.0
    model.eval()
    for (i, (inputs, labels)) in enumerate(val_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        # Forward pass. (Prediction stage)
        # _, scores = model(inputs)
        # scores = torch.stack([scores[:, k - 1] for k in [1, 2, 4, 5, 6, 7, 15, 17, 18, 19, 20, 21, 22, 23, 24, 29, 56, 63, 64, 69]], dim = 1)
        # print(scores.shape)

        scores = model(inputs)
        
        cum_loss += loss_fn(scores, labels).item()

        # Count how many correct in this batch.
        max_labels = scores > 0
        correct += sum([f1_score(a, b) for a, b in zip(max_labels.detach().cpu(), labels.detach().cpu())])

    val_accuracies.append(correct / len(valset))
    val_losses.append(cum_loss / (i + 1))

    # Logging the current results on validation.
    print('Avg-Loss: %.4f, Accuracy: %.4f' % 
        (cum_loss / (i + 1), correct / len(valset)))

def predict(model, batchSize, valset):

    # We don't need to bach the validation set but let's do it anyway.
    val_loader = torch.utils.data.DataLoader(dataset = valset,
                                                batch_size = batchSize,
                                                shuffle = False) # No need.

    val_predict = []
    val_names = []

    # GPU enabling.
    model = model.cuda()

    # Make a pass over the validation data.
    model.eval()
    for i, (name, inputs) in enumerate(val_loader):
        inputs = inputs.cuda()

        # Forward pass. (Prediction stage)
        _, scores = model(inputs)
        scores = torch.stack([scores[:, k - 1] for k in [1, 2, 4, 5, 6, 7, 15, 17, 18, 19, 20, 21, 22, 23, 24, 29, 56, 63, 64, 69]], dim = 1)
        # print(scores.shape)

        # scores = model(inputs)

        # Count how many correct in this batch.
        max_labels = scores > 0

        val_predict.append(max_labels)
        val_names.append(name)
      
    return val_predict, val_names

# !rm -rf L2G; git clone https://github.com/PengtaoJiang/L2G.git

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('L2G')
import models
m = torch.load('/content/drive/MyDrive/Colab Notebooks/CS701/resnet38_coco.pt')

m = torch.load('/content/drive/MyDrive/Colab Notebooks/CS701/resnet50_cs701.pt')

# Pytorch DataLoader for iterating over batches.
batchSize = 8

# Create the model.
loss_fn = nn.BCEWithLogitsLoss()

predicts, names = predict(m, batchSize, valset)

valset[0][1].min()

names = [x for n in names for x in n]

doc = ''
for s, p in enumerate(torch.cat(predicts, 0)):
    doc = doc + names[s] + ' '
    for c, pp in enumerate(p):
        if pp:
            doc = doc + str(c) + ' '
    doc = doc + '\n'

d2 = sorted(doc.split('\n'))

print('\n'.join(d2))

print(doc)

# Pytorch DataLoader for iterating over batches.
batchSize = 8

# Create the model.
loss_fn = nn.BCEWithLogitsLoss()

eval_model(m, loss_fn, batchSize, valset)

def view(x):
    x.T

import matplotlib.pyplot as plt

plt.imshow(valset[0][0].T)

valset[0][1]

outputs = m(valset[0][0].unsqueeze(0).cuda())

outputs[0].shape

(outputs[1] > 0).long()