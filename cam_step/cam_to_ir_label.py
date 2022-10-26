import os
import numpy as np
import imageio

from torch import multiprocessing
from torch.utils.data import DataLoader

from pipeline.dataloader import VOC12ImageDataset, decode_int_filename
from utils import torchutils, imutils


def _work(process_id, infer_dataset, args):

    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)

    for iter, pack in enumerate(infer_data_loader):
        img_name = decode_int_filename(pack['name'][0])
        img = pack['img'][0].numpy()
        cam_dict = np.load(os.path.join(args.cam_out_dir, img_name + '.npy'), allow_pickle=True).item()

        cams = cam_dict['high_res']
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        # 1. find confident fg & bg
        fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_fg_thres)
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        if keys.shape[0]==1:
            conf = np.full(fg_conf_cam.shape, 0)
        else:
            pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
            fg_conf = keys[pred]

            bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_bg_thres)
            bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
            pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
            bg_conf = keys[pred]

            # 2. combine confident fg & bg
            conf = fg_conf.copy()
            #conf[fg_conf != 0] = fg_conf
            conf[bg_conf + fg_conf == 0] = 0

        imageio.imwrite(os.path.join(args.ir_label_out_dir, img_name + '.png'),
                        conf.astype(np.uint8))


        if process_id == args.num_workers - 1 and iter % (len(databin) // 20) == 0:
            print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')

def run(args):
    file = "data/public/val.json"
    dataset = VOC12ImageDataset(file, "public/val_image/", img_normal=None, to_torch=False)
    dataset = torchutils.split_dataset(dataset, 8)
    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args), join=True)
    print(']')
