import os
import json
import argparse
import numpy as np
import xml.dom.minidom as XML



voc_cls_id = open('../cs701/public/class.txt')

def transdifi(data_path):
    print("generating final json file for VOC07 dataset")
    train_label_dir = data_path + "/train_label.txt"
    val_label_dir = data_path + '/submission/label_.txt'
    train_img_dir = data_path + "/train_image/"
    val_img_dir = data_path + "/val_image/"

    train_data = []
    for k, v in {x.strip().split()[0]:[int(k) for k in x.strip().split()[1:]] for x in open(train_label_dir, 'r').readlines()}.items():
        img_path = train_img_dir + k
        target = [int(i in v) for i in range(80)]
        train_data.append({'target':target, 'img_path':img_path})

    val_data = []
    for k, v in {x.strip().split()[0]:[int(k) for k in x.strip().split()[1:]] for x in open(val_label_dir, 'r').readlines()}.items():
        img_path = val_img_dir + k
        target = [int(i in v) for i in range(80)]
        val_data.append({'target':target, 'img_path':img_path})




    json.dump(train_data, open("data/public/train.json", "w"))
    json.dump(val_data, open("data/public/val.json", "w"))
    print("public data preparing finished!")
    # print("data/voc07/trainval_voc07.json data/voc07/test_voc07.json")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Usage: --data_path /your/dataset/path/VOCdevkit
    parser.add_argument("--data_path", default="../cs701/public", type=str, help="The absolute path of cs701")
    args = parser.parse_args()

    if not os.path.exists("data/public"):
        os.makedirs("data/public")

    transdifi(args.data_path)