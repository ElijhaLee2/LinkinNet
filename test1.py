from other.config import SAVE_STEP, TOTAL_EPOCH, CUDA_VISIBLE_DEVICES, IS_RESTORE, \
    N_DIS, SAVE_STEP_EPOCH, DISPLAY_STEP, IS_STACK_0, IS_STACK_1, WORK_DIR_NAME, BATCH_SIZE, ALLOW_GROWTH
import os
import sys
import time
from other.config import IMG_LENGTH, BATCH_SIZE, N_CAT, CAT_NUMs
from pycocotools.coco import COCO
from other.data_path import CAP_PATH, INSTANCES_PATH, IMG_PATH, SEG_PATH
import numpy as np
import h5py
from collections import defaultdict
import skimage.io as io

# os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

out_path = './test_res'

# coco
coco = COCO(INSTANCES_PATH)
coco_caps = COCO(CAP_PATH)

# img_list
img_list = os.listdir(IMG_PATH)

# prepare feed
embs = []
segs = []
caps = []
for i in range(BATCH_SIZE):
    index = int(np.random.randint(0, len(img_list), []))
    img_name = img_list[index]

    # img_name --> seg_name --> seg
    seg_name = img_name.replace('.jpg', '.png')
    seg = io.imread(os.path.join(SEG_PATH, seg_name))
    segs.append(seg)

    # img_name --> img_no --> ann --> captions
    img_no = int(img_name[15: -4])
    cap_list = coco_caps.imgToAnns[img_no]
    cap_index = int(np.random.randint(0, 5, []))
    cap = cap_list[cap_index]
    caps.append(cap['caption'])

f = open(os.path.join(out_path, 'caps.txt'), 'w')
for cap in caps:
    f.writelines(cap + '\n')
f.flush()
f.close()
