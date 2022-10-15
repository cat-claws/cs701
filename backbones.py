from torchvision import models
import torch.nn as nn

def get_model(name):
	if name == 'resnet101':
		m = models.resnet101(weights="IMAGENET1K_V1")
		m.fc = nn.Linear(in_features=2048, out_features=20, bias=True)

	return m