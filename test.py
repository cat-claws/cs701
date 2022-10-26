import torch
from PIL import Image
from pipeline import resnet_csra
from pipeline.resnet_cam import CAM
from pipeline.resnet_irn import EdgeDisplacement
import numpy as np
import torch
from pipeline.dataloader import VOC12ClassificationDatasetMSF
from torch.utils.data import DataLoader
# from pipeline.dataloader import load_image_label_list_from_npy

# stat_dict = torch.load("pipeline/resnet101_coco_head4_lam0.5_83.3.pth", torch.device("cpu"))

# m = CAM(4, 0.5, 80)

# m.load_state_dict(stat_dict)

# a = np.zeros((1,3,448,448))
# i = np.stack([a, np.flip(a, -1)], axis=0)
# print(i.shape)
# # # m.eval()
# m = m.double()
# print(m(a).size())
model = EdgeDisplacement()
a = torch.ones((2,3,463,640))
b1,b2 = model(a)
print(b2.size())
