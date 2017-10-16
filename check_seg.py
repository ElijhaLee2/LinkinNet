import skimage.io as io
import numpy as np
import os
from data_input.read_data import read_picked_data
from other.data_path import PKL_PATH, SEG_PATH, IMG_PATH

name_list, embedding_list, caption_list = read_picked_data(PKL_PATH)

for i in range(20):
    # prepare feed
    index = int(np.random.randint(0, len(name_list), []))
    # img_name --> img
    img_name = name_list[index]
    # img = io.imread(os.path.join(IMG_PATH, img_name))

    # img_name --> seg_name --> seg
    seg_name = img_name.replace('.jpg', '.png')
    seg = np.expand_dims(io.imread(os.path.join(SEG_PATH, seg_name)), -1)

    tmp = np.reshape(seg, [-1]).tolist()
    seg_set = set(tmp)

    print(img_name)
    print(seg_set)
