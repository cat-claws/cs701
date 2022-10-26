import torch
from PIL import Image
from pipeline import resnet_csra
from pipeline.resnet_cam import CAM
import numpy as np
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
file = "data/public/val.json"
dataset = VOC12ClassificationDatasetMSF(file,
                         voc12_root="data/public/val_image/", scales=(1.0, 0.5, 1.5, 2.0))
data_loader = data_loader = DataLoader(dataset, shuffle=False, pin_memory=False)
for iter, pack in enumerate(data_loader):
    img_name = pack['name'][0]
    label = pack['label']
    size = pack['size']
    
    print(img_name)
    print(label)
    print(size)
    print(pack['img'][0])
    break