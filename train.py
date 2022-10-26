import utils
import backbones

import glob
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms

m = backbones.get_model('deeplab')
m = m.to('cuda' if torch.cuda.is_available() else 'cpu')

from sklearn.metrics import f1_score, accuracy_score
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, loss_fn, batchSize, trainset, valset, optimizer, num_epochs):
  
    # Shuffling is needed in case dataset is not shuffled by default.
    train_loader = torch.utils.data.DataLoader(dataset = trainset,
                                                batch_size = batchSize,
                                                shuffle = True,
                                                num_workers= 4)
    # We don't need to bach the validation set but let's do it anyway.
    val_loader = torch.utils.data.DataLoader(dataset = valset,
                                            batch_size = batchSize,
                                            shuffle = False,
                                            num_workers= 4) # No need.

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
batchSize = 4

# Create the model.
loss_fn = nn.BCEWithLogitsLoss()

# Define a learning rate. 
learningRate = 1e-3

# Optimizer.
optimizer = torch.optim.Adam(m.parameters(), lr = learningRate)
train_model(m, loss_fn, batchSize, utils.trainset_, utils.valset_, optimizer, 50)

torch.save(m, 'csra_resnet50.pt')
