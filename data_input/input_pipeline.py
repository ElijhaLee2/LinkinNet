import time
import os
import numpy as np

# from pycocotools.coco import COCO

import tensorflow as tf
from data_input.read_data import read_picked_data
from other.data_path import IMG_PATH, SEG_PATH, PKL_PATH, INSTANCES_PATH
from other.config import IMG_LENGTH, BATCH_SIZE, N_CAT, CAT_NUMs
from other.function import preprocess, select_seg


def get_input_tensors(image_root=IMG_PATH, seg_root=SEG_PATH,
                      image_length=IMG_LENGTH, batch_size=BATCH_SIZE):
    print('Start reading training data...')

    # coco = COCO(INSTANCES_PATH)
    # obtain image_name_list, embedding_tensor_list(每个embedding_tensor有5个embedding), caption_list
    # name_list, embedding_list, caption_list = read_data(coco, catNms=CAT_NMS)
    name_list, embedding_list, caption_list = read_picked_data(PKL_PATH)
    name_list_shuffle = np.random.permutation(name_list)
    length = len(name_list)

    # obtain the a slice of emb, emb_WRONG, caption
    embedding_tensor = tf.convert_to_tensor(embedding_list)
    caption_tensor = tf.convert_to_tensor(caption_list)

    [embedding_slice, caption_slice] = get_emb_cap(embedding_tensor, caption_tensor)

    index = tf.random_uniform(shape=(), maxval=5, dtype=tf.int32)
    emb = embedding_slice[index]
    cap = caption_slice[index]

    # 用两个reader分别读取img和seg
    img_path_list = [os.path.join(image_root, image_file) for image_file in name_list]
    img_path_list_w = [os.path.join(image_root, image_file) for image_file in name_list_shuffle]
    img_path_queue = tf.train.string_input_producer(img_path_list, shuffle=False, capacity=batch_size * 4,
                                                    name='img_path_queue')
    img_path_queue_w = tf.train.string_input_producer(img_path_list_w, shuffle=False, capacity=batch_size * 4,
                                                      name='img_path_queue_w')

    seg_path_list = [os.path.join(seg_root, image_file.replace('.jpg', '.png')) for image_file in name_list]
    seg_path_list_w = [os.path.join(seg_root, image_file.replace('.jpg', '.png')) for image_file in name_list_shuffle]
    seg_path_queue = tf.train.string_input_producer(seg_path_list, shuffle=False, capacity=batch_size * 4,
                                                    name='seg_path_queue')
    seg_path_queue_w = tf.train.string_input_producer(seg_path_list_w, shuffle=False, capacity=batch_size * 4,
                                                      name='seg_path_queue_w')


    _, img = tf.WholeFileReader().read(img_path_queue)
    _, img_w = tf.WholeFileReader().read(img_path_queue_w)
    _, seg = tf.WholeFileReader().read(seg_path_queue)
    _, seg_w = tf.WholeFileReader().read(seg_path_queue_w)

    # decode, flip, one_hot, resize, crop
    img, seg = preprocess(tf.image.decode_jpeg(img, channels=3),
                          tf.image.decode_png(seg, channels=1),
                          image_length)

    img_w, seg_w = preprocess(tf.image.decode_jpeg(img_w, channels=3),
                              tf.image.decode_png(seg_w, channels=1),
                              image_length)
    # TODO0: after pre & one_hot
    # a1 = tf.cast(tf.one_hot(seg, N_CAT)[:, :, 0, :], tf.float32)
    # ss0 = tf.transpose(tf.expand_dims(a1, -2), [3, 0, 1, 2])
    # tf.summary.image('seg_pre', ss0, 100)

    # seg = select_seg(seg, CAT_NUMs)
    # seg_w = select_seg(seg_w, CAT_NUMs)

    # # TODO0: after choose (should be no problem,
    # # because 'choose_seg' affect channel-level information instead of pixel-level
    # ss = tf.transpose(tf.expand_dims(seg, -2), [3, 0, 1, 2])
    # tf.summary.image('seg_choose', ss, 100)

    # 打包成shuffle batch
    batch = tf.train.shuffle_batch([img, img_w, seg, seg_w, emb, cap],
                                   batch_size=batch_size, capacity=batch_size * 5,
                                   min_after_dequeue=batch_size * 4)

    print('Reading finished.')
    return batch, length


def get_emb_cap(embs, caps):
    # def is_same_index(in1, in2):
    #     return tf.equal(in1, in2)
    #
    # def next_index(in1, in2):
    #     return in1, queue_wrong.dequeue()

    range_size = embs.get_shape().as_list()[0]
    queue = tf.train.range_input_producer(range_size, shuffle=False)
    # queue_wrong = tf.train.range_input_producer(range_size, shuffle=True)

    index1 = queue.dequeue()
    # index2 = queue_wrong.dequeue()

    # input of 'cond', input of 'body' and loop_vars must be same!
    # _, index2 = tf.while_loop(is_same_index, next_index, (index1, index2))

    # return tf.gather(embs, index1), tf.gather(embs, index2), tf.gather(caps, index1)
    return tf.gather(embs, index1), tf.gather(caps, index1)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # time.sleep(2)
    # embs = tf.convert_to_tensor([1., 2., 3., 4.], dtype=tf.float32)
    # # embs = tf.placeholder(tf.float32, shape=[1000,5,1024])
    # caps = tf.convert_to_tensor(['a', 'b', 'c', 'd'], dtype=tf.string)
    # te, we, cap = get_emb_cap(embs, caps)
    #
    # coord = tf.train.Coordinator()
    # with tf.Session() as sess:
    #     tf.train.start_queue_runners(sess, coord)
    #     try:
    #         while 1:
    #             res = sess.run([te, we, cap])
    #             assert te != we, 'te==we!!'
    #             print(res)
    #     finally:
    #         coord.request_stop()
    get_input_tensors()