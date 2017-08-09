import os
from pycocotools.coco import COCO

import tensorflow as tf
from data_input.read_data import read_cat_datas
from other.data_path import IMG_PATH, SEG_PATH, HDF5_PATH, CAP_PATH, INSTANCES_PATH
from other.config import IMG_SIZE, BATCH_SIZE, IS_DEBUG,N_CLASS


def get_input_tensors(hdf5_path=HDF5_PATH, ann_path = CAP_PATH, image_root=IMG_PATH, seg_root=SEG_PATH, image_size=IMG_SIZE, is_debug=IS_DEBUG, batch_size=BATCH_SIZE):
    print('Start reading training data...')

    coco = COCO(INSTANCES_PATH)
    # 1.得到image_name_list和embedding_tensor_list(每个embedding_tensor有5个embedding)
    name_list, embedding_list, caption_list = read_cat_datas(coco, supNms=['person'])
    length = len(name_list)

    # 2. embedding
    embedding_tensor = tf.convert_to_tensor(embedding_list)
    caption_tensor = tf.convert_to_tensor(caption_list)
    [embedding_slice, caption_slice] = tf.train.slice_input_producer([embedding_tensor, caption_tensor],shuffle=False,capacity=BATCH_SIZE*4)
    # Produce wrong embedding
    [wrong_embedding_slice] = tf.train.slice_input_producer([embedding_tensor],shuffle=True,capacity=BATCH_SIZE*4)

    # def is_emb_same(embedding_slice,wrong_embedding_slice):
    #     return tf.equal(embedding_slice,wrong_embedding_slice)
    #
    # def next_emb(embedding_slice,wrong_embedding_slice):
    #     return real_k,wrong_k_new, wrong_v_new
    # _, wrong_k, wrong_v = tf.while_loop(is_emb_same, next_image, (real_k, wrong_k, wrong_v))

    index = tf.random_uniform(shape=(),maxval=5,dtype=tf.int32)
    emb = embedding_slice[index]
    cap = caption_slice[index]
    wrong_emb = wrong_embedding_slice[index]

    # 3.用两个reader分别读取img和seg
    img_path_list = [os.path.join(image_root, image_file) for image_file in name_list]
    img_path_queue = tf.train.string_input_producer(img_path_list, shuffle=False, capacity=batch_size * 4,
                                                               name='img_path_queue')
    seg_path_list = [os.path.join(seg_root,image_file[0:-4]+'.png') for image_file in name_list]
    seg_path_queue = tf.train.string_input_producer(seg_path_list, shuffle=False, capacity=batch_size * 4,
                                                                   name='seg_path_queue')
    reader_img = tf.WholeFileReader()
    reader_seg = tf.WholeFileReader()
    img_k,img_v = reader_img.read(img_path_queue)
    seg_k,seg_v = reader_seg.read(seg_path_queue)

    # 解码图片
    img = tf.image.resize_images(tf.image.decode_jpeg(img_v,channels=3)/255,(image_size,image_size))
    seg = tf.one_hot(   # 把segmentation转化为one-hot形式
        tf.image.resize_images(tf.image.decode_jpeg(seg_v,channels=1),(image_size,image_size),method=1)[:,:,0],  # 必须保持像素值为整数
        N_CLASS
    )

    # 4.
    # 打包成shuffle batch
    batch = tf.train.shuffle_batch([img, seg, emb, wrong_emb, cap],
                                   batch_size=batch_size, capacity=batch_size * 5,
                                   min_after_dequeue=batch_size * 4)

    print('Reading finished.')
    return batch, length

