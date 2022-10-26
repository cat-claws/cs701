import argparse
import os

import sys
sys.path.append('utils')
import pyutils
# from utils import pyutils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", required=True, type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset # Here is for the training data
    # parser.add_argument("--train_list", default="../segmentation/data/datasets/food_public/train_aug.txt", type=str)
    # parser.add_argument("--val_list", default="../segmentation/data/datasets/food_public/train_aug.txt", type=str)
    # parser.add_argument("--infer_list", default="../segmentation/data/datasets/food_public/train_aug.txt", type=str,
    #                     help="voc12/train_aug.txt to train a fully supervised model, "
    #                          "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")


    # here is for the validation/test datasets
    parser.add_argument("--train_list", default="data/public/val.json", type=str)
    parser.add_argument("--val_list", default="data/public/val.json", type=str)
    parser.add_argument("--infer_list", default="data/public/val.json", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")


    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="pipeline.resnet_cam", type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=32, type=int)
    parser.add_argument("--cam_num_epoches", default=20, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.05, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="pipeline.resnet_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=2, type=int)
    parser.add_argument("--irn_num_epoches", default=10, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.25)
    parser.add_argument("--sem_seg_bg_thres", default=0.25)

    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="sess/res101_cam.pth", type=str)
    parser.add_argument("--irn_weights_name", default="sess/res101_irn.pth", type=str)
    parser.add_argument("--cam_out_dir", default="result/cam", type=str)
    parser.add_argument("--ir_label_out_dir", default="result/ir_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="result/sem_seg", type=str)
    parser.add_argument("--ins_seg_out_dir", default="result/ins_seg", type=str)

    # Step
    parser.add_argument("--train_cam_pass", default=False)
    parser.add_argument("--make_cam_pass", default=False)
    parser.add_argument("--eval_cam_pass", default=False)
    parser.add_argument("--cam_to_ir_label_pass", default=True)
    parser.add_argument("--train_irn_pass", default=True)
    parser.add_argument("--make_ins_seg_pass", default=True)
    parser.add_argument("--eval_ins_seg_pass", default=False)
    parser.add_argument("--make_sem_seg_pass", default=True)
    parser.add_argument("--eval_sem_seg_pass", default=False)

    args = parser.parse_args()

    os.makedirs("sess", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    os.makedirs(args.ins_seg_out_dir, exist_ok=True)

    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    if args.train_cam_pass is True:
        import cam_step.train_cam

        timer = pyutils.Timer('step.train_cam:')
        cam_step.train_cam.run(args)

    if args.make_cam_pass is True:
        import cam_step.make_cam

        timer = pyutils.Timer('step.make_cam:')
        cam_step.make_cam.run(args)


    if args.cam_to_ir_label_pass is True:
        import cam_step.cam_to_ir_label_origin

        timer = pyutils.Timer('step.cam_to_ir_label:')
        cam_step.cam_to_ir_label_origin.run(args)

    if args.train_irn_pass is True:
        import cam_step.train_irn

        timer = pyutils.Timer('step.train_irn:')
        cam_step.train_irn.run(args)

    if args.make_ins_seg_pass is True:
        import cam_step.make_ins_seg_labels

        timer = pyutils.Timer('step.make_ins_seg_labels:')
        cam_step.make_ins_seg_labels.run(args)


    if args.make_sem_seg_pass is True:
        import cam_step.make_sem_seg_labels

        timer = pyutils.Timer('step.make_sem_seg_labels:')
        cam_step.make_sem_seg_labels.run(args)
