import glob
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms

class SquarePad:
	def __call__(self, image):
		_, w, h = image.shape
		max_wh = 640
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (vp, max_wh - h - vp, hp, max_wh - w - hp)
		return F.pad(image, padding, 'constant')

class CS701Dataset(torch.utils.data.Dataset):
	def __init__(self, split, transform=None):
		images = {s.split('/')[-1]:np.array(Image.open(s)) for s in glob.glob(f"public/{split}_image/*.jpg")}
		if split == 'train':
			labels = {x.strip().split()[0]:[int(k) for k in x.strip().split()[1:]] for x in open('public/train_label.txt', 'r').readlines()}
		elif split == 'val':
			labels = {k:[0] for k in images}
		self.transform = transform
		self.data = [(k, v, labels[k])for k, v in images.items()]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):

		name, image, label = self.data[index]
		label = sum([F.one_hot(torch.tensor(k), 20) for k in label]).float()

		if self.transform:
			image = self.transform(image)
		return name, image, label

# Load MNIST dataset
transform=transforms.Compose([
	transforms.ToTensor(),
	SquarePad(),
	transforms.RandomHorizontalFlip(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

trainset = CS701Dataset('train', transform=transform)
valset = CS701Dataset('val', transform=transform)

torch.manual_seed(1234)
trainset, valset = torch.utils.data.random_split(trainset, [13003, 1000])
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
# val_loader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=True)