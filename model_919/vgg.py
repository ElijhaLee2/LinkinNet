import tensorflow as tf
import scipy.io
import numpy as np
from other.config import IMG_LENGTH
from other.data_path import VGG_MODEL_PATH


class VGG:
    def __init__(self, input_, seg=None, name='VGG', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            self.input = tf.identity(input_)
            if seg is not None:
                self.seg = tf.identity(seg)
            net = {}
            vgg_rawnet = scipy.io.loadmat(VGG_MODEL_PATH)
            vgg_layers = vgg_rawnet['layers'][0]
            build_net = self.build_net
            get_weight_bias = self.get_weight_bias
            # 64
            net['input'] = self.input - np.array([123.6800, 116.7790, 103.9390]).reshape((1, 1, 1, 3))
            net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(vgg_layers, 0), name='vgg_conv1_1')
            net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2), name='vgg_conv1_2')
            net['pool1'] = build_net('pool', net['conv1_2'])
            # 32
            net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5), name='vgg_conv2_1')
            net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7), name='vgg_conv2_2')
            net['pool2'] = build_net('pool', net['conv2_2'])
            # 16
            net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10), name='vgg_conv3_1')
            net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12), name='vgg_conv3_2')
            net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14), name='vgg_conv3_3')
            net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16), name='vgg_conv3_4')
            net['pool3'] = build_net('pool', net['conv3_4'])
            # 8
            net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19), name='vgg_conv4_1')
            net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21), name='vgg_conv4_2')
            net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23), name='vgg_conv4_3')
            net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25), name='vgg_conv4_4')
            net['pool4'] = build_net('pool', net['conv4_4'])
            # 4

            net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28), name='vgg_conv5_1')
            net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30), name='vgg_conv5_2')
            # net['conv5_3'] = build_net('conv', net['conv5_2'], get_weight_bias(vgg_layers, 32), name='vgg_conv5_3')
            # net['conv5_4'] = build_net('conv', net['conv5_3'], get_weight_bias(vgg_layers, 34), name='vgg_conv5_4')
            # net['pool5'] = build_net('pool', net['conv5_4'])
            # # 2
            self.net = net

    def build_net(self, ntype, nin, nwb=None, name=None):
        if ntype == 'conv':
            return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) + nwb[1])
        elif ntype == 'pool':
            return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def get_weight_bias(self, vgg_layers, i):
        weights = vgg_layers[i][0][0][0][0][0]
        weights = tf.constant(weights)
        bias = vgg_layers[i][0][0][0][0][1]
        bias = tf.constant(np.reshape(bias, bias.size))
        return weights, bias


def compute_vgg_loss(vgg_real, vgg_fake, seg):
    def compute_error(real, fake, seg):
        return tf.reduce_mean(
            seg * tf.reduce_mean(tf.abs(fake - real), axis=-1, keep_dims=True),
            reduction_indices=[1, 2])

    vgg_real = vgg_real.net
    vgg_fake = vgg_fake.net
    # 64
    p0 = compute_error(vgg_real['input'], vgg_fake['input'], seg)
    # 64
    p1 = compute_error(vgg_real['conv1_2'], vgg_fake['conv1_2'], seg) / 1.6
    # 32
    p2 = compute_error(vgg_real['conv2_2'], vgg_fake['conv2_2'],
                       tf.image.resize_nearest_neighbor(seg, (IMG_LENGTH // 2, IMG_LENGTH // 2))) / 2.3
    # 16
    p3 = compute_error(vgg_real['conv3_2'], vgg_fake['conv3_2'],
                       tf.image.resize_nearest_neighbor(seg, (IMG_LENGTH // 4, IMG_LENGTH // 4))) / 1.8
    # 8
    p4 = compute_error(vgg_real['conv4_2'], vgg_fake['conv4_2'],
                       tf.image.resize_nearest_neighbor(seg, (IMG_LENGTH // 8, IMG_LENGTH // 8))) / 2.8
    # 4
    p5 = compute_error(vgg_real['conv5_2'], vgg_fake['conv5_2'],
                       tf.image.resize_nearest_neighbor(seg, (IMG_LENGTH // 16, IMG_LENGTH // 16))) / 0.08
    # content_loss = p0 + p1 + p2 + p3 + p4 + p5
    content_loss = p1 + p2 + p3 + p4 + p5  # [BATCH_SIZE,N_CAT]
    vgg_loss = tf.reduce_sum(content_loss)
    return vgg_loss
