import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
from pipeline.dataloader import VOC12ClassificationDatasetMSF
from torchvision.models.segmentation import deeplabv3_resnet50
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from PIL import Image
import numpy as np
import os

from utils import torchutils

cudnn.enabled = True

class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model): 
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        output = self.model(x)
        if isinstance(output, torch.Tensor):
            return output
        else:
            return output["out"]
    
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        print(model_output)
        return (model_output[self.category, :, : ] * self.mask).sum()

def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=8 // n_gpus, pin_memory=False)

    with cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']
            image = np.array(Image.open(pack['img_path'][0]))
            rgb_img = np.float32(image) / 255
            input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            input_tensor = input_tensor.cuda()
            model_for_out = SegmentationModelOutputWrapper(model)
            output = model_for_out(input_tensor)
            normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
            valid_label = torch.nonzero(label)[:,0]
            targets = []
            for l in valid_label:
                label_value = int(l)
                mask = normalized_masks[0,:,:,:].argmax(axis=0).detach().cpu().numpy()
                mask_uint8 = 255 * np.uint8(mask == l)
                mask_float = np.float32(mask == label_value)
                Image.fromarray(both_images)
                targets.append(SemanticSegmentationTarget(label_value, mask_float))
            target_layers = [model.backbone.layer4]
            cam = GradCAM(model=model, target_layers=target_layers,
                        use_cuda=torch.cuda.is_available())
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                    # cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_list = np.array(grayscale_cam)
            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_label, "cam": cam_list, "high_res": cam_list})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = deeplabv3_resnet50(pretrained=True)
    model.eval()

    n_gpus = torch.cuda.device_count()
    
    file = "data/public/val.json"

    dataset = VOC12ClassificationDatasetMSF(file,
                         voc12_root="public/val_image/", scales=(1.0, 0.5, 1.5, 2.0))
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()
