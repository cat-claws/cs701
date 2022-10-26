import utils
import backbones

import glob
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms

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
    for i, (name, inputs, _) in enumerate(val_loader):
        inputs = inputs.cuda()

        # Forward pass. (Prediction stage)
        # _, scores = model(inputs)
        # scores = torch.stack([scores[:, k - 1] for k in [1, 2, 4, 5, 6, 7, 15, 17, 18, 19, 20, 21, 22, 23, 24, 29, 56, 63, 64, 69]], dim = 1)
        # print(scores.shape)

        scores = model(inputs)

        # Count how many correct in this batch.
        max_labels = scores > 0

        val_predict.append(max_labels)
        val_names.append(name)
      
    return val_predict, val_names



m = torch.load('checkpoints/resnet_cs701_29.pt')
batchSize = 4

loss_fn = nn.BCEWithLogitsLoss()

predicts, names = predict(m, batchSize, utils.valset)

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

with open('public/submission/label.txt', 'w') as f:
	f.write('\n'.join(d2))
