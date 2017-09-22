import tensorflow as tf
import skimage.io as io
import time, os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
time.sleep(2)

img_size = 64
img_dir = '/data/bo718.wang/zhaowei/data/516data/mscoco/train2014/mask/COCO_train2014_000000291677.png'

L = 255
CL = 255


def _get_mask(img, threshold):
    low = threshold[0]
    high = threshold[1]
    mask_low = tf.cast(tf.greater_equal(img, low), tf.int32)
    mask_high = tf.cast(tf.less(img, high), tf.int32)
    mask = mask_low * mask_high
    return mask


def _trans_l14(img, mask):
    shape = tf.shape(img)
    r = tf.zeros(shape, tf.int32)
    g = img * 4
    b = CL * tf.ones(shape, tf.int32)
    return tf.concat([r, g, b], axis=-1) * mask


def _trans_l24(img, mask):
    shape = tf.shape(img)
    r = tf.zeros(shape, tf.int32)
    g = CL * tf.ones(shape, tf.int32)
    b = 4 * (CL // 2 - img)
    return tf.concat([r, g, b], axis=-1) * mask


def _trans_l34(img, mask):
    shape = tf.shape(img)
    r = 4 * (img - CL // 2)
    g = tf.zeros(shape, tf.int32)
    b = CL * tf.ones(shape, tf.int32)
    return tf.concat([r, g, b], axis=-1) * mask


def _trans_l44(img, mask):
    shape = tf.shape(img)
    r = CL * tf.ones(shape, tf.int32)
    g = 4 * (CL - img)
    b = tf.zeros(shape, tf.int32)
    return tf.concat([r, g, b], axis=-1) * mask


def trans(img):
    assert type(img) == tf.Tensor, '\'img\' must be a tf.Tensor, got %s' % (type(img))
    len_ = len(img.get_shape().as_list())
    assert len_ == 4, 'The length of \'img\' must be 4, got %d' % len_
    l14, l24, l34, l44 = L // 4, 2 * L // 4, 3 * L // 4, 4 * L // 4
    img = tf.cast(img, tf.int32)
    img = img * 255 // 91
    # mask_bg = _trans_bg(img)
    mask_l14 = _trans_l14(img, _get_mask(img, [1, l14]))
    mask_l24 = _trans_l24(img, _get_mask(img, [l14 + 1, l24]))
    mask_l34 = _trans_l34(img, _get_mask(img, [l24 + 1, l34]))
    mask_l44 = _trans_l44(img, _get_mask(img, [l34 + 1, l44]))

    return tf.cast(mask_l14 + mask_l24 + mask_l34 + mask_l44, tf.uint8)


if __name__ == '__main__':
    seg = io.imread(img_dir)
    seg = np.expand_dims(seg, 0)
    seg = np.expand_dims(seg, -1)
    img = io.imread((img_dir.replace('.png','.jpg')).replace('mask','train2014'))

    img_ph = tf.placeholder(tf.int8, shape=[None, None, None, None])
    trans_img = trans(img_ph)

    with tf.Session() as sess:
        trans_res = sess.run(trans_img, {img_ph: seg})
        io.imsave('./hhh.png', np.concatenate([trans_res[0],img],axis=1))
