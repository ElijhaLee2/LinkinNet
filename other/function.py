import tensorflow as tf
import tensorflow.contrib.layers as ly
import os
import shutil
from other.config import N_DIS, N_CAT, BATCH_SIZE
import numpy as np


# def leaky_relu(x, leak=0.2, name="lrelu"):
#     return tf.maximum(x, leak * x)


# def lrelu(x, leak=0.2, name="lrelu"):
#         f1 = 0.5 * (1 + leak)
#         f2 = 0.5 * (1 - leak)
#         return f1 * x + f2 * abs(x)

# def make_multi_dirs(path_list):
#     for path in path_list:
#         os.makedirs(path)

def rms(tensor_list):
    sum = tf.Variable(0., trainable=False)
    len = tf.Variable(0., trainable=False)
    for t in tensor_list:
        # When tensor_list is a list of gradients, t is None means the corresponding variable is independent with the output
        if t is None:
            continue
        sum += tf.reduce_sum(tf.square(t))
        len += tf.reduce_prod(tf.convert_to_tensor(t.get_shape().as_list(), tf.float32))
    return tf.sqrt(sum / len)


def backup_model_file(backup_dir):
    # shutil.copytree('./model', os.path.join(backup_dir, 'model'))
    # os.makedirs(os.path.join(backup_dir, 'other'), exist_ok=True)
    # shutil.copy('./other/hyperparameter.py', os.path.join(backup_dir, 'other', 'hyperparameter.py'))
    # shutil.copy('./other/config.py', os.path.join(backup_dir, 'other', 'config.py'))
    # shutil.copy('./train.py', os.path.join(backup_dir, 'train.py'))
    for dir_name in os.listdir('.'):
        if dir_name in ['params', 'preprocess', 'tests', 'useless', '__pycache__', 'work_dir', 'liwei','test_res']:
            continue

        dir_absolute = os.path.join('.', dir_name)
        if os.path.isdir(dir_absolute):
            shutil.copytree(dir_absolute, os.path.join(backup_dir, dir_name))
        if os.path.isfile(dir_absolute):
            shutil.copyfile(dir_absolute, os.path.join(backup_dir, dir_name))


def tile(emb, pic_size: list):
    mul = pic_size[0] * pic_size[1]
    [batch_size, channel] = emb.get_shape().as_list()

    res = [emb] * mul
    res = tf.stack(res, axis=1)
    res = tf.reshape(res, [batch_size, pic_size[0], pic_size[1], channel])

    return res


# def get_trained_num(gs):
#     if gs > 30:
#         res = (30 * N_DIS[0] + (gs - 30) * N_DIS[1]) * BATCH_SIZE
#     else:
#         res = (gs * N_DIS[0]) * BATCH_SIZE
#
#     return res


def preprocess(img, seg, length: int):
    def random_flip(img, seg):
        rand_ud = tf.random_uniform([], 0, 1.0)
        rand_lr = tf.random_uniform([], 0, 1.0)
        cond_ud = tf.less(rand_ud, .5)
        cond_lr = tf.less(rand_lr, .5)

        res_img_ud = tf.cond(cond_ud,
                             lambda: tf.reverse(img, [0]),
                             lambda: img)
        res_seg_ud = tf.cond(cond_ud,
                             lambda: tf.reverse(seg, [0]),
                             lambda: seg)

        res_img_lr = tf.cond(cond_lr,
                             lambda: tf.reverse(res_img_ud, [1]),
                             lambda: res_img_ud)
        res_seg_lr = tf.cond(cond_lr,
                             lambda: tf.reverse(res_seg_ud, [1]),
                             lambda: res_seg_ud)

        return res_img_lr, res_seg_lr

    def random_crop(img, seg, size):
        # h,w of img and seg must be the same!
        shape1 = tf.shape(img)
        # shape2 = tf.shape(seg)
        # channel_img = img.get_shape().as_list()[-1]
        # channel_seg = seg.get_shape().as_list()[-1]

        size1 = tf.convert_to_tensor(size + [3])
        size2 = tf.convert_to_tensor(size + [1])
        limit = shape1 - size1 + 1

        offset = tf.random_uniform(tf.shape(shape1), dtype=tf.int32, maxval=tf.int32.max) % limit
        return tf.slice(img, offset, size1), tf.slice(seg, offset, size2)

    # 1. resize
    resize_shape = tf.cast([1.15 * length, 1.15 * length], tf.int32)
    # img: uint8 --> float32
    img_resize = tf.image.resize_area(tf.expand_dims(img, 0), resize_shape)[0]
    # seg: uint8 --> uint8
    seg_resize = tf.image.resize_nearest_neighbor(tf.expand_dims(seg, 0), resize_shape)[0]

    # 2. crop
    img_crop, seg_crop = random_crop(img_resize, seg_resize, [length, length])

    # 3. flip
    img_flip, seg_flip = random_flip(img_crop, seg_crop)

    # 4. img.normalize, seg.one_hot
    img_normalized = img_flip / 255
    seg_one_hot = tf.one_hot(seg_flip, N_CAT, axis=-1, dtype=tf.float32)[:, :, 0, :]

    # crop
    return img_normalized, seg_one_hot


# # wrong order for resize, causing low efficiency
# def preprocess(img, seg, length: int):
#     def random_flip(img, seg):
#         concat = tf.concat([img, seg], axis=-1)
#         ret = tf.image.random_flip_left_right(concat)
#         ret = tf.image.random_flip_up_down(ret)
#         return tf.split(ret, [3, 1], axis=2)
#
#     def random_crop(img, seg, size):
#         # h,w of img and seg must be the same!
#         shape1 = tf.shape(img)
#         # shape2 = tf.shape(seg)
#         channel_img = img.get_shape().as_list()[-1]
#         channel_seg = seg.get_shape().as_list()[-1]
#
#         size1 = tf.convert_to_tensor([size[0], size[1], channel_img])
#         size2 = tf.convert_to_tensor([size[0], size[1], channel_seg])
#         limit = shape1 - size1 + 1
#
#         offset = tf.random_uniform(tf.shape(shape1), dtype=tf.int32, maxval=tf.int32.max) % limit
#         return tf.slice(img, offset, size1), tf.slice(seg, offset, size2)
#
#     # shape = tf.cast(tf.shape(img), tf.float32)
#     # min_edge = tf.cast(tf.minimum(shape[0], shape[1]), tf.int32)
#
#     # flip
#     img_flip, seg_flip = random_flip(img, seg)
#
#     # seg -- one hot
#     seg_one_hot = tf.one_hot(seg_flip, N_CAT)[:, :, 0, :]  # length, length,N_CAT
#
#     # resize (to 1.15*length)
#     resize_shape = tf.cast([1.15 * length, 1.15 * length], tf.int32)
#     img_resize = tf.image.resize_area(tf.expand_dims(img_flip, 0),
#                                       resize_shape)[0] / 255  # 0~1. . 'resize_area' will convert to float32.
#     seg_resize = tf.image.resize_area(tf.expand_dims(seg_one_hot, 0), resize_shape)[0]  # 0/1
#
#     # crop (to length)
#     # crop_shape = tf.convert_to_tensor([length, length, 4])
#     img_crop, seg_crop = random_crop(img_resize, seg_resize,
#                                      [length, length])
#
#     return img_crop, seg_crop  # seg_ has been uint8 all the time


def l2_norm_square(t):
    dim = len(t.get_shape().as_list())
    reduce = list(range(1, dim))
    return tf.reduce_sum(tf.square(t), reduction_indices=reduce)


def select_seg(one_hot_seg, chosen_cat: list):
    def select_fg(fg, class_no: list):
        channel_first = tf.transpose(fg, [2, 0, 1])
        # channel_no = []
        # for c in class_no:
        #     channel_no += [c]
        fg_chosen = tf.gather(channel_first, class_no)
        return tf.transpose(fg_chosen, [1, 2, 0])

    one_hot_seg = tf.cast(tf.identity(one_hot_seg, 'one_hot_seg'), tf.float32)
    length = one_hot_seg.get_shape().as_list()[0]
    # chosen_cat.append(0)
    chosen_cat_set = set(chosen_cat)
    all_cat_set = set(np.arange(0, N_CAT))
    unchosen_cat_set = all_cat_set - chosen_cat_set

    chosen_bool = tf.reduce_sum(tf.one_hot(tf.convert_to_tensor(np.array(list(chosen_cat_set))), N_CAT), 0)
    unchosen_bool = tf.reduce_sum(tf.one_hot(tf.convert_to_tensor(np.array(list(unchosen_cat_set))), N_CAT), 0)
    tmp = tf.tile(
        tf.expand_dims(tf.expand_dims(tf.concat([chosen_bool, unchosen_bool], axis=0), 0), 0),
        [length, length, 1])
    [chosen_bool, unchosen_bool] = tf.split(tmp, 2, 2)

    bg = tf.reduce_sum(unchosen_bool * one_hot_seg, -1, True)
    fg = (chosen_bool * one_hot_seg)  # bg included
    # choose selected class
    fg = select_fg(fg, chosen_cat)  # bg not included
    res = tf.concat([bg, fg], axis=-1)  # bg in fg is excluded
    return res


def summarize(group, sum_name, tensor, type, collec='LinkNet_sum'):
    with tf.name_scope(None):  # Use no prefix for summary, easy to observe in tensorboard
        if type == 'scl':
            tf.summary.scalar(group + '/' + sum_name, tensor, collections=[collec])
        elif type == 'img':
            tf.summary.image(group + '/' + sum_name, tensor, BATCH_SIZE, collections=[collec])
        elif type == 'his':
            tf.summary.histogram(group + '/' + sum_name, tensor, collections=[collec])
        else:
            raise TypeError('No such summary type!: ', type)


def remain_time(time_interval, total_epoch, length_epoch, batch_size, global_step, n_dis):
    mul = n_dis[0] / n_dis[1]
    if global_step < 30:
        return time_interval * (total_epoch * length_epoch / batch_size - global_step) / mul / (3600 * 24)  # days
    else:
        return time_interval * (total_epoch * length_epoch / batch_size - global_step) / (3600 * 24)  # days


if __name__ == '__main__':
    x = tf.placeholder(tf.uint8, [64, 64, 91])
    select_seg(x, [18])
