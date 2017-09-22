import time
import os, h5py
import numpy as np

import tensorflow as tf
from data_input.read_data import read_data
from other.config import IMG_LENGTH, BATCH_SIZE, N_CAT, CAT_NMS
from other.function import preprocess_flower

IMG_PATH = "/data/rui.wu/irfan/text-to-image-master/Data/flowers/jpg/"
HDF5_PATH = "/data/rui.wu/irfan/text-to-image-master/Data/flowers/flower_tv.hdf5"


# CAP_PATH = "/data/rui.wu/irfan/text-to-image-master/Data/flowers/text_c10"

def get_input_tensors():
    print('Start reading training data...')

    emb_hdf5 = h5py.File(HDF5_PATH)
    length = len(emb_hdf5)

    name_list = list()
    emb_list = list()
    for k, v in emb_hdf5.items():
        name_list.append(k)
        emb_list.append((np.array(v)-0.00036593081)/0.020409096)

    embedding_tensor = tf.convert_to_tensor(np.array(emb_list))

    emb_slice = tf.train.slice_input_producer([embedding_tensor], shuffle=False)
    wrong_emb_slice = tf.train.slice_input_producer([embedding_tensor], shuffle=True, capacity=BATCH_SIZE * 5)

    index = tf.random_uniform(shape=(), maxval=5, dtype=tf.int32)
    emb = emb_slice[0][index]
    wrong_emb = wrong_emb_slice[0][index]

    name_queue = tf.train.string_input_producer([os.path.join(IMG_PATH,name) for name in name_list],shuffle=False)

    # read img
    img_k, img_v = tf.WholeFileReader().read(name_queue)

    # decode, crop, flip, resize
    img = preprocess_flower(tf.image.decode_jpeg(img_v, channels=3), IMG_LENGTH)

    # pack into shuffle batch
    batch = tf.train.shuffle_batch([img, emb, wrong_emb],
                                   batch_size=BATCH_SIZE, capacity=BATCH_SIZE * 5,
                                   min_after_dequeue=BATCH_SIZE * 3)
    print('Reading finished.')
    return batch, length


if __name__ == '__main__':
    get_input_tensors()
