import h5py
import os
import numpy as np
import json
from other.config import BATCH_SIZE
import json
from collections import defaultdict
# import tensorflow as tf

def read_hdf5(hdf5_path, ann_path, is_debug):
    h = h5py.File(hdf5_path)
    c = read_captions(ann_path)
    # ann_list = os.listdir(ann_path)
    # ann_ids = []
    # [ann_ids.append(int(id[15:-4])) for id in ann_list]


    names = []
    embeddings = [] # 每个image有5条embedding
    captions = []

    h_items = h.items()
    cnt = 0
    total = BATCH_SIZE if is_debug else len(h_items)
    for hh in h_items:
        # id = int(hh[0][15:-4])
        # if ann_ids.count(id)<1:
        #     continue
        names.append(hh[0])
        embeddings.append(np.array(hh[1])[0:5]) # 注意！有一个图片对应了6个caption，导致封装np.array的时候出错！
        captions.append(c[hh[0]][0:5])
        cnt+=1
        if cnt>=total:
            print('read_hdf5 finished. Total: %d'%cnt)
            break
    return names, np.array(embeddings), captions

# JSON_PATH = "/data/bo718.wang/+zhaowei/data/516data/mscoco/annotations/captions_train2014.json"
# SEG_PATH = "/data/bo718.wang/zhaowei/data/516data/mscoco/train2014/mask"
# IMG_PATH = "/data/bo718.wang/zhaowei/data/516data/mscoco/train2014/train2014"
# HDF5_PATH = "/data/rui.wu/CZHH/Dataset_COCO/COCO_VSE_torch/COCO_vse_torch_train.hdf5"

# read_hdf5(HDF5_PATH,JSON_PATH,True)


def read_captions(json_path):
    image_captions = defaultdict(list)
    with open(json_path) as f:
        ic_data = json.load(f)
        for idx in range (0, len(ic_data['annotations'])):
            img_path = 'COCO_%s2014_%.12d.jpg'%('train', ic_data['annotations'][idx]['image_id'])
            image_captions[img_path].append(ic_data['annotations'][idx]['caption'])

    return image_captions