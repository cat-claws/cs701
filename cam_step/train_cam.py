import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from pipeline.dataset import DataSet

import importlib


from utils import pyutils, torchutils


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['target'].cuda(non_blocking=True)

            x = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss1': loss1.item()})
            model.train()

        print('loss: %.4f' % (val_loss_meter.pop('loss1')))

        return



def run(args):

    model = getattr(importlib.import_module(args.cam_network), 'Net')()

    # model
    # if args.model == "resnet101": 
    #     model = resnet_cam.Net()     
    model.cuda()
    if torch.cuda.device_count() > 1:
        print("lets use {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # data
    train_file = ["data/public/train.json"]
    test_file = ["data/public/val.json"]

    train_dataset = DataSet(train_file, args.train_aug, args.img_size, args.dataset)
    val_dataset = DataSet(test_file, args.test_aug, args.img_size, args.dataset)
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size, shuffle=True, num_workers=8)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size, shuffle=False, num_workers=8)

    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            label = pack['target'].cuda(non_blocking=True)
            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label.float())

            avg_meter.add({'loss1': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        else:
            # validate(model, val_data_loader)
            print("Don't Validate.")
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
    torch.cuda.empty_cache()