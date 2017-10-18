import tensorflow as tf
import os
import sys
import time
from model_919 import Generator
from other.config import IMG_LENGTH, N_CAT
# from pycocotools.coco import COCO
from other.data_path import PKL_PATH, SEG_PATH, IMG_PATH, HDF5_PATH
import numpy as np
import h5py
import skimage.io as io
from data_input.read_data import read_picked_data
from other.function import preprocess
from data_input.input_pipeline import get_input_tensors


# batch, length = get_input_tensors(batch_size=1)
# [img, img_w, seg, seg_w, emb, _] = batch

# TODO change restore_path to .../0919xxxx/save
restore_path = "/home/elijha/PycharmProjects/LinkinNet/LinkinNet-params/0919"
out_path = os.path.join('.', 'test_res', time.strftime("%m-%d_%H-%M-%S"))
if not os.path.isdir(out_path):
    os.makedirs(out_path)
sample_total_num = 100

# placeholders
emb_ph = tf.placeholder(tf.float32, [1, 1024], 'emb_ph')
img_ph = tf.placeholder(tf.uint8, [None, None, 3], 'img_ph')
seg_ph = tf.placeholder(tf.uint8, [None, None, 1], 'seg_ph')
img_ph_, seg_ph_ = preprocess(img_ph, seg_ph, IMG_LENGTH)
img_ph_, seg_ph_ = tf.expand_dims(img_ph_, 0), tf.expand_dims(seg_ph_, 0)
# emb_ph = emb
# img_ph_ = img
# seg_ph_ = seg

# build graph
gen = Generator.Generator(emb_ph, 'gen_img', batch_size=1, seg=seg_ph_, reuse=False)

# Summary
sum_img = tf.summary.image('sum_img', tf.concat([img_ph_, gen.generated_pic], axis=2))
merge = tf.summary.merge([sum_img])

# saver
saver = tf.train.Saver()
cp = tf.train.latest_checkpoint(restore_path)

# coord
# coord = tf.train.Coordinator()

# File writer
fw = tf.summary.FileWriter(logdir=out_path)

# data
name_list, embedding_list, caption_list = read_picked_data(PKL_PATH)

# caption target
f = open(os.path.join(out_path, 'caps.txt'), 'w')
# Session restore & run
sess = tf.InteractiveSession()
saver.restore(sess, cp)
# tf.global_variables_initializer().run()

# tf.train.start_queue_runners(sess, coord)

# Start sample
for i in range(sample_total_num):
    # prepare feed
    index = int(np.random.randint(0, len(name_list), []))

    # img_name --> img
    img_name = name_list[index]
    img = io.imread(os.path.join(IMG_PATH, img_name))

    # img_name --> seg_name --> seg
    seg_name = img_name.replace('.jpg', '.png')
    seg = np.expand_dims(io.imread(os.path.join(SEG_PATH, seg_name)), -1)

    # # prepare feed
    # index = int(np.random.randint(0, 5, []))
    #
    # seg_list = os.listdir(SEG_PATH)
    # seg_name = seg_list[index]
    # seg = np.expand_dims(io.imread(os.path.join(SEG_PATH, seg_name)), -1)
    #
    # img_name = seg_name.replace('.png', '.jpg')
    # img = io.imread(os.path.join(IMG_PATH, img_name))

    # captions
    cap_index = int(np.random.randint(0, 5, []))
    cap = caption_list[index][cap_index]

    # emb
    emb_index = cap_index
    emb = np.expand_dims(embedding_list[index][emb_index], 0)
    # h = h5py.File(HDF5_PATH)
    # emb = np.expand_dims(np.array(h[img_name])[emb_index], 0)

    feed = {
        img_ph: img,
        seg_ph: seg,
        emb_ph: emb,
    }

    # Session run
    [real_img, seg_, fake_img, merge_] = sess.run([img_ph_, seg_ph_, gen.generated_pic, merge], feed_dict=feed)
    # [real_img, seg_, fake_img, merge_] = sess.run([img_ph_, seg_ph_, gen.generated_pic, merge])
    fw.add_summary(merge_)

    # Save results
    io.imsave(os.path.join(out_path, str(i) + '.jpg'), np.uint8(np.vstack([real_img[0], fake_img[0]]) * 255.))
    io.imsave(os.path.join(out_path, str(i) + '_.png'), np.uint8(np.argmax(seg_[0], 2)))

    tmp = np.reshape(np.uint8(np.argmax(seg_[0], 2)), [-1]).tolist()
    seg_set = set(tmp)
    print(seg_set)
    # print(np.max(real_img), ',', np.min(real_img))
    # print(np.max(fake_img), ',', np.min(fake_img))
    # io.imsave(os.path.join(out_path, str(i) + '.jpg'), np.hstack([real_img[0], fake_img[0]]) % 1.)
    # io.imsave(os.path.join(out_path, str(i) + '.jpg'), fake_img[0] * 2 - 1)
    # f.writelines(str(i) + '. ' + cap + '\n')

# Finish
f.flush()
f.close()
fw.close()
# coord.request_stop()
sess.close()
