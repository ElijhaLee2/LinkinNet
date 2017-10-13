import tensorflow as tf
from model_919.Trainer import Trainer
from data_input.input_pipeline import get_input_tensors
from other.config import SAVE_STEP, TOTAL_EPOCH, CUDA_VISIBLE_DEVICES, IS_RESTORE, \
    N_DIS, SAVE_STEP_EPOCH, DISPLAY_STEP, IS_STACK_0, IS_STACK_1, WORK_DIR_NAME, BATCH_SIZE, ALLOW_GROWTH
from other.function import backup_model_file, remain_time
import os
import sys
import time
from model_919.whole import build_whole_graph
from model_919 import Generator
from other.config import IMG_LENGTH, BATCH_SIZE, N_CAT, CAT_NUMs
from pycocotools.coco import COCO
from other.data_path import CAP_PATH, INSTANCES_PATH, IMG_PATH, HDF5_PATH, SEG_PATH
import numpy as np
import h5py
from data_input.read_hdf5 import read_hdf5
from collections import defaultdict
import skimage.io as io

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

restore_path = ''
out_path = './test_res'


# placeholders
emb_ph = tf.placeholder(tf.float32, [BATCH_SIZE, 1024], 'emb_ph')
seg_ph = tf.placeholder(tf.uint8, [BATCH_SIZE, IMG_LENGTH, IMG_LENGTH, 1], 'seg_ph')

# build graph
gen = Generator(emb_ph, 'gen_img', seg=seg_ph, reuse=False)

# saver
saver = tf.train.Saver()
cp = tf.train.latest_checkpoint(restore_path)

# coco
coco = COCO(INSTANCES_PATH)
coco_caps = COCO(CAP_PATH)

# HDF5 file
h = h5py.File(HDF5_PATH)

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

    # img_name --> emb
    emb_index = cap_index
    emb = np.array(h[img_name])[emb_index]
    embs.append(emb)

# feed:
feed = {
    emb_ph: np.stack(embs, 0),
    seg_ph: np.stack(segs, 0),
}

# Session & restore & run
sess = tf.InteractiveSession()
saver.restore(sess, cp)
fake_img = sess.run(gen.generated_pic, feed_dict=feed)

f = open(os.path.join(out_path, 'caps.txt'), 'w')
for cap in caps:
    f.writelines(cap + '\n')
f.flush()
f.close()
