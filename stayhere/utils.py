import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms

from pycocotools.coco import COCO

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_data_public(split):
	images = os.listdir(f"public/{split}_image")
	if split == 'train':
		labels = {x.strip().split()[0]:[int(k) for k in x.strip().split()[1:]] for x in open('public/train_label.txt', 'r').readlines()}
	elif split == 'val':
		labels = {k:[0] for k in images}
	data = [(os.path.join(f"public/{split}_image", k), labels[k]) for k in images]
	return data


def coco_to_cs701(classes_1, classes_2, labels):
	new_labels = set()
	for c in labels:
		if classes_2[c] in classes_1:
			new_labels.add(classes_1[classes_2[c]])
	return list(new_labels)

	
def build_data_coco(root, annFile):
	coco = COCO(annFile)
	ids = list(sorted(coco.imgs.keys()))
	data = [(os.path.join(root, coco.loadImgs(k)[0]["file_name"]), list({ins['category_id'] for ins in coco.loadAnns(coco.getAnnIds(k))})) for k in ids]
	cs701_classes = {c:k for k, c in enumerate(pd.read_csv('public/class.txt', header = None, delimiter=r"\s+")[1])}
	coco_classes = pd.read_csv('coco/labels.txt', header = None)[0]
	data = [(d[0], coco_to_cs701(cs701_classes, coco_classes, d[1])) for d in data]
	data = [d for d in data if d[1] != []]
	return data


class SquarePad:
	def __call__(self, image):
		_, w, h = image.shape
		max_wh = 640
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (vp, max_wh - h - vp, hp, max_wh - w - hp)
		return F.pad(image, padding, 'constant')

class CS701Dataset(torch.utils.data.Dataset):
	def __init__(self, data, transform=None):
		self.data = data
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):

		image_path, label = self.data[index]
		image = Image.open(image_path).convert("RGB")
		name = image_path.split('/')[-1]

		label = sum([F.one_hot(torch.tensor(k), 20) for k in label]).float()

		if self.transform:
			image = self.transform(image)

		return name, image, label

# Load MNIST dataset
train_transform=transforms.Compose([
	transforms.ToTensor(),
	SquarePad(),
	transforms.Resize(224),
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform=transforms.Compose([
	transforms.ToTensor(),
	SquarePad(),
	transforms.Resize(224),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


coco_train = build_data_coco("coco/train2017", "coco/annotations/instances_train2017.json")
coco_val = build_data_coco("coco/val2017", "coco/annotations/instances_val2017.json")

public_train = build_data_public('train')
public_val = build_data_public('val')

trainset = CS701Dataset(public_train, transform=train_transform)
# trainset = CS701Dataset('train', transform=train_transform)
valset = CS701Dataset(public_val, transform=val_transform)

torch.manual_seed(1234)
trainset_, valset_ = torch.utils.data.random_split(trainset, [len(trainset) - 1000, 1000])
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
# val_loader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=True)