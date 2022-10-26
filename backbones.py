from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(name):
	if name == 'resnet101':
		m = models.resnet101(weights="IMAGENET1K_V1")
		m.fc = nn.Linear(in_features=2048, out_features=20, bias=True)
		# for name, param in m.named_parameters():                
		# 	if name.startswith('conv1') or name.startswith('bn1') or name.startswith('layer1') or name.startswith('layer2'):
		# 		param.requires_grad = False

	elif name == 'csra_resnet50':
		m = ResNet_CSRA(num_heads=1, lam=0.1, num_classes=20)

	elif name == 'deeplab':
		m = Modified(models.segmentation.deeplabv3_resnet101(models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1))

	print(f'The model has {count_parameters(m):,} trainable parameters')
	return m

from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock


class CSRA(nn.Module): # one basic block 
	def __init__(self, input_dim, num_classes, T, lam):
		super(CSRA, self).__init__()
		self.T = T      # temperature       
		self.lam = lam  # Lambda                        
		self.head = nn.Conv2d(input_dim, num_classes, 1, bias=False)
		self.softmax = nn.Softmax(dim=2)

	def forward(self, x):
		# x (B d H W)
		# normalize classifier
		# score (B C HxW)
		score = self.head(x) / torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0,1)
		score = score.flatten(2)
		base_logit = torch.mean(score, dim=2)

		if self.T == 99: # max-pooling
			att_logit = torch.max(score, dim=2)[0]
		else:
			score_soft = self.softmax(score * self.T)
			att_logit = torch.sum(score * score_soft, dim=2)

		return base_logit + self.lam * att_logit

    


class MHA(nn.Module):  # multi-head attention
	temp_settings = {  # softmax temperature settings
		1: [1],
		2: [1, 99],
		4: [1, 2, 4, 99],
		6: [1, 2, 3, 4, 5, 99],
		8: [1, 2, 3, 4, 5, 6, 7, 99]
	}

	def __init__(self, num_heads, lam, input_dim, num_classes):
		super(MHA, self).__init__()
		self.temp_list = self.temp_settings[num_heads]
		self.multi_head = nn.ModuleList([
			CSRA(input_dim, num_classes, self.temp_list[i], lam)
			for i in range(num_heads)
		])

	def forward(self, x):
		logit = 0.
		for head in self.multi_head:
			logit += head(x)
		return logit


import torch.utils.model_zoo as model_zoo
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}





class ResNet_CSRA(ResNet):
	arch_settings = {
		18: (BasicBlock, (2, 2, 2, 2)),
		34: (BasicBlock, (3, 4, 6, 3)),
		50: (Bottleneck, (3, 4, 6, 3)),
		101: (Bottleneck, (3, 4, 23, 3)),
		152: (Bottleneck, (3, 8, 36, 3))
	}

	def __init__(self, num_heads, lam, num_classes, depth=50, input_dim=2048, cutmix=None):
		self.block, self.layers = self.arch_settings[depth]
		self.depth = depth
		super(ResNet_CSRA, self).__init__(self.block, self.layers)
		self.init_weights(pretrained=True, cutmix=cutmix)

		self.classifier = MHA(num_heads, lam, input_dim, num_classes) 
		self.loss_func = F.binary_cross_entropy_with_logits

	def backbone(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		
		return x

	def forward(self, x):
		x = self.backbone(x)
		x = self.classifier(x)
		return x

	def init_weights(self, pretrained=True, cutmix=None):
		if cutmix is not None:
			print("backbone params inited by CutMix pretrained model")
			state_dict = torch.load(cutmix)
		elif pretrained:
			print("backbone params inited by Pytorch official model")
			model_url = model_urls["resnet{}".format(self.depth)]
			state_dict = model_zoo.load_url(model_url)

		model_dict = self.state_dict()
		try:
			pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
			self.load_state_dict(pretrained_dict)
		except:
			logger = logging.getLogger()
			logger.info(
				"the keys in pretrained model is not equal to the keys in the ResNet you choose, trying to fix...")
			state_dict = self._keysFix(model_dict, state_dict)
			self.load_state_dict(state_dict)

		# remove the original 1000-class fc
		self.fc = nn.Sequential() 


class FeatureExtractor(nn.Module):
	def __init__(self, resnet_model):
		super(FeatureExtractor, self).__init__()
		# remove the last layer
		self.truncated_resnet = nn.Sequential(*list(resnet_model.children())[:-1])
	def forward(self, x):
		feats = self.truncated_resnet(x)
		return feats#.view(feats.size(0), -1)

class Modified(nn.Module):
	def __init__(self, pretrained_model):
		super(Modified, self).__init__()
		self.pretrained_model = pretrained_model
	def forward(self, x):
		out = self.pretrained_model(x)['out']
		_, _, h, w = out.shape
		pred = F.avg_pool2d(out[:, :-1], kernel_size=(h, w), padding=0)
		pred = pred.view(pred.size(0), -1)
		return pred

