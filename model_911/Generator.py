import tensorflow as tf

from model_911.hyperparameter import GENERATOR_HP
from other.config import BATCH_SIZE, N_CAT
from other.function import tile
from other.nn_func import conv, upsample_conv_with_concat, upsample_conv, dense, deconv
from model_911.model_func import residual_block
import numpy as np


class Generator:
    def __init__(self, embedding, name: str, seg=None, reuse=False):
        with tf.name_scope(name) as scope:
            self.embedding = tf.identity(embedding)
            if name.find('seg') != -1:
                assert seg is None
                self.generated_pic = self._build_0(self.embedding, name, reuse)
            elif name.find('img') != -1:
                assert type(seg) == tf.Tensor
                self.seg = tf.identity(seg)
                self.generated_pic = self._build_1(self.seg, self.embedding, name, reuse)
            else:
                raise ValueError('Nether \'seg\' nor \'img\' is in \'name\' of Generator')

            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            self.model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

            # self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,scope=scope))
            # tf.summary.text('caption', caption)
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)

    def _build_0(self, embedding, name, reuse):
        z_dim = GENERATOR_HP['z_dim']
        base_channel = GENERATOR_HP['base_channel']
        activation_fn = GENERATOR_HP['activation_fn']
        normalizer_fn = GENERATOR_HP['normalizer_fn']

        with tf.variable_scope(name, reuse=reuse) as scope:
            z = tf.random_normal([BATCH_SIZE, z_dim])
            input_ = tf.concat([z, embedding], axis=1)
            net = dense(input_, 8 * 8 * 4 * base_channel, None, activation_fn, 0)

            net = residual_block(net, 4 * base_channel, normalizer_fn, activation_fn, 1)
            net = residual_block(net, 4 * base_channel, normalizer_fn, activation_fn, 3)
            net = residual_block(net, 4 * base_channel, normalizer_fn, activation_fn, 5)
            net = residual_block(net, 4 * base_channel, normalizer_fn, activation_fn, 7)

            # 64, 91
            generated_img = self.decode_seg(net, base_channel, normalizer_fn, activation_fn, 9)
        return generated_img

    def _build_1(self, seg, emb, name, reuse):
        gf_dim = GENERATOR_HP['gf_dim']
        act_fn = GENERATOR_HP['act_fn']
        norm_fn = GENERATOR_HP['norm_fn']
        z_dim = GENERATOR_HP['z_dim']

        with tf.variable_scope(name, reuse=reuse) as scope:
            [_, length, _, c] = seg.get_shape().as_list()
            size = np.array([length, length])
            seg_16 = tf.image.resize_area(seg, size // 16)
            z = tf.random_normal([BATCH_SIZE, z_dim], name='z')
            emb_z = tf.concat([emb, z], axis=-1, name='emb_z_concat')
            net = tf.concat([tile(emb_z, list(size // 16)), seg_16], axis=-1)
            net = conv(net, gf_dim * 8, 1, 1, None, act_fn, 0)
            net = conv(net, gf_dim * 8, 3, 1, norm_fn, None, 1)  # ATTENTION: NO ACT_FN!
            # s16,8*gf

            # residual blocks
            net = residual_block(net, gf_dim * 8, norm_fn, act_fn, 2)
            net = self.upsample(net, tf.image.resize_area(seg, size // 8), norm_fn, 5)
            # s8,4*gf
            net = residual_block(net, gf_dim * 4, norm_fn, act_fn, 6)
            net = self.upsample(net, tf.image.resize_area(seg, size // 4), norm_fn, 9)
            # s4,2*gf
            net = residual_block(net, gf_dim * 2, norm_fn, act_fn, 10)
            net = self.upsample(net, tf.image.resize_area(seg, size // 2), norm_fn, 13)
            # s2,gf
            net = residual_block(net, gf_dim * 1, norm_fn, act_fn, 14)
            net = self.upsample(net, tf.image.resize_area(seg, size // 1), norm_fn, 17)
            # s,gf/2
            net = residual_block(net, gf_dim * 0.5, norm_fn, act_fn, 18)
            net = conv(act_fn(net), 3, 3, 1, None, tf.tanh, 21)
            # s,3.  combine channels into 3, generating a pic
            generated_img = net / 2 + 0.5

            return generated_img

    # def encode_pic(self, image_input, base_channel, normalizer_fn, activation_fn):
    #     # s = 64
    #     net_s = image_input
    #     net_s2 = conv(image_input, base_channel, 4, 2, None, activation_fn, 0)
    #     net_s4 = conv(net_s2, 2 * base_channel, 4, 2, normalizer_fn, activation_fn, 1)
    #     net_s8 = conv(net_s4, 4 * base_channel, 4, 2, normalizer_fn, activation_fn, 2)
    #     # To obtain a better generated fake pic, just down-sample the segmentation to s8
    #
    #     return net_s8, [net_s, net_s2, net_s4]

    # def wrap_emb(self, emb, tile_size):
    #     return tile(emb, tile_size)



    def upsample(self, feature_map, label, norm_fn, ly_index):
        [_, img_length, _, c] = feature_map.get_shape().as_list()
        fm_up = tf.image.resize_bilinear(feature_map, [img_length * 2, img_length * 2])
        concat = tf.concat([fm_up, label], axis=-1)
        # ATTENTION: res is NOT ACT!
        res = conv(concat, c // 2, 3, 1, norm_fn, None, ly_index)
        return res

    # def decode_img(self, input_, base_channel, net_ss, normalizer_fn, activation_fn, start_index):
    #     [net_s, net_s2, net_s4] = net_ss
    #     # with tf.variable_scope('decode_pic'):
    #     net = upsample_conv_with_concat(input_, net_s4, base_channel, 3, 1, normalizer_fn, activation_fn, start_index)
    #     net = conv(net, base_channel, 3, 1, normalizer_fn, activation_fn, start_index + 1)
    #     net = conv(net, base_channel, 3, 1, normalizer_fn, activation_fn, start_index + 2)
    #     net = upsample_conv_with_concat(net, net_s2, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 3)
    #     net = conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 4)
    #     net = conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 5)
    #     net = upsample_conv_with_concat(net, net_s, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 6)
    #     net = conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 7)
    #     net = conv(net, 3, 4, 1, None, tf.tanh, start_index + 8)
    #
    #     # return net / 2 + 0.5
    #     return net

    def decode_seg(self, input_, base_channel, normalizer_fn, activation_fn, start_index):
        net = upsample_conv(input_, base_channel, 3, 1, normalizer_fn, activation_fn, start_index)
        net = conv(net, base_channel, 3, 1, normalizer_fn, activation_fn, start_index + 1)
        net = conv(net, base_channel, 3, 1, normalizer_fn, activation_fn, start_index + 2)
        net = upsample_conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 3)
        net = conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 4)
        net = conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 5)
        net = upsample_conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 6)
        net = conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 7)
        net = conv(net, N_CAT, 4, 1, None, None, start_index + 8)

        return net / 2 + 0.5


if __name__ == '__main__':
    emb = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1024])
    # cap = tf.placeholder(tf.string, shape=[BATCH_SIZE, 1])
    seg = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 64, 64, N_CAT])
    g = Generator(emb, 'g_img', seg=seg)
    print()
