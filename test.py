import tensorflow as tf
import os
import sys
import time
from model_919 import Generator
from other.config import IMG_LENGTH, N_CAT
from pycocotools.coco import COCO
from other.data_path import CAP_PATH, INSTANCES_PATH, IMG_PATH, HDF5_PATH, SEG_PATH
import numpy as np
import h5py
import skimage.io as io

# os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

restore_path = "/home/elijha/PycharmProjects/LinkinNet/LinkinNet-params/0919"
out_path = os.path.join('.', 'test_res', time.strftime("%m-%d_%H-%M-%S"))
if not os.path.isdir(out_path):
    os.makedirs(out_path)
sample_total_num = 30


def preprocess(seg, length: int):
    def random_flip(seg):
        rand_lr = tf.random_uniform([], 0, 1.0)
        cond_lr = tf.less(rand_lr, .5)

        res_seg_lr = tf.cond(cond_lr,
                             lambda: tf.reverse(seg, [1]),
                             lambda: seg)

        return res_seg_lr

    def random_crop(seg, size):
        # h,w of img and seg must be the same!
        shape1 = tf.shape(seg)
        # shape2 = tf.shape(seg)
        # channel_img = img.get_shape().as_list()[-1]
        # channel_seg = seg.get_shape().as_list()[-1]

        size1 = tf.convert_to_tensor(size + [1])
        limit = shape1 - size1 + 1

        offset = tf.random_uniform(tf.shape(shape1), dtype=tf.int32, maxval=tf.int32.max) % limit
        return tf.slice(seg, offset, size1)

    # 1. resize
    resize_shape = tf.cast([1.15 * length, 1.15 * length], tf.int32)
    # seg: uint8 --> uint8
    seg_resize = tf.image.resize_nearest_neighbor(tf.expand_dims(seg, 0), resize_shape)[0]

    # 2. crop
    seg_crop = random_crop(seg_resize, [length, length])

    # 3. flip
    seg_flip = random_flip(seg_crop)

    # 4. seg.one_hot
    seg_one_hot = tf.one_hot(seg_flip, N_CAT, axis=-1, dtype=tf.float32)[:, :, 0, :]

    # crop
    return tf.expand_dims(seg_one_hot, 0)


# placeholders
emb_ph = tf.placeholder(tf.float32, [1, 1024], 'emb_ph')
seg_ph = tf.placeholder(tf.uint8, [None, None, 1], 'seg_ph')
seg_ph_ = preprocess(seg_ph, IMG_LENGTH)

# build graph
gen = Generator.Generator(emb_ph, 'gen_img', batch_size=1, seg=seg_ph_, reuse=False)

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

# caption target
f = open(os.path.join(out_path, 'caps.txt'), 'w')
# Session restore & run
sess = tf.InteractiveSession()
saver.restore(sess, cp)
# tf.global_variables_initializer().run()

# Start sample
for i in range(sample_total_num):
    # prepare feed
    index = int(np.random.randint(0, len(img_list), []))
    img_name = img_list[index]

    # img_name --> seg_name --> seg
    seg_name = img_name.replace('.jpg', '.png')
    seg = np.expand_dims(io.imread(os.path.join(SEG_PATH, seg_name)), -1)
    # seg = seg
    # segs.append(seg)

    # img_name --> img_no --> ann --> captions
    img_no = int(img_name[15: -4])
    cap_list = coco_caps.imgToAnns[img_no]
    cap_index = int(np.random.randint(0, 5, []))
    cap = cap_list[cap_index]['caption']

    # img_name --> emb
    emb_index = cap_index
    emb = np.expand_dims(np.array(h[img_name])[emb_index], 0)

    feed = {
        emb_ph: emb,
        seg_ph: seg,
    }

    # Session run
    fake_img = sess.run(gen.generated_pic, feed_dict=feed)

    # Save results
    io.imsave(os.path.join(out_path, str(i) + '.jpg'), np.uint8(fake_img[0] * 255))
    # io.imsave(os.path.join(out_path, str(i) + '.jpg'), fake_img[0] * 2 - 1)
    f.writelines(str(i) + '. ' + cap + '\n')

# Finish
f.flush()
f.close()
sess.close()
