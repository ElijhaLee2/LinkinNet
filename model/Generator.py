import tensorflow as tf

from model.hyperparameter import GENERATOR_HP
from other.config import BATCH_SIZE, N_CAT
from other.function import tile
from other.nn_func import conv, upsample_conv_with_concat, upsample_conv, dense


class Generator:
    def __init__(self, embedding, caption, name: str, seg=None, reuse=False):
        with tf.name_scope(name) as scope:
            if name.find('seg') != -1:
                assert seg is None
                self.generated_pic = self._build_0(embedding, name, reuse)
            elif name.find('img') != -1:
                assert type(seg) == tf.Tensor
                self.seg = tf.identity(seg)
                self.generated_pic = self._build_1(self.seg, self.embedding, name, reuse)
            else:
                raise ValueError('Nether \'seg\' nor \'img\' is in \'name\' of Generator')

            self.embedding = tf.identity(embedding)
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            self.model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

            # self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,scope=scope))
            tf.summary.text('caption', caption)
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

            net = self.residual_block(net, 4 * base_channel, normalizer_fn, activation_fn, 1)
            net = self.residual_block(net, 4 * base_channel, normalizer_fn, activation_fn, 3)
            net = self.residual_block(net, 4 * base_channel, normalizer_fn, activation_fn, 5)
            net = self.residual_block(net, 4 * base_channel, normalizer_fn, activation_fn, 7)

            # 64, 91
            generated_img = self.decode_seg(net, base_channel, normalizer_fn, activation_fn, 9)
        return generated_img

    def _build_1(self, seg, emb, name, reuse):
        base_channel = GENERATOR_HP['base_channel']
        activation_fn = GENERATOR_HP['activation_fn']
        normalizer_fn = GENERATOR_HP['normalizer_fn']

        with tf.variable_scope(name, reuse=reuse) as scope:
            # 64, N_CLASS
            encoded_pic, net_ss = self.encode_pic(seg, base_channel, normalizer_fn, activation_fn)  # s8
            shape = encoded_pic.get_shape().as_list()
            tiled_emb = tile(emb, [shape[1], shape[2]])
            concat = tf.concat([encoded_pic, tiled_emb], axis=3)  # (4 + 16)* bc

            net = conv(concat, 8 * base_channel, 1, 1, None,  # normalizer_fn is None 'cause emb is INPUT
                       activation_fn, '3_concat')
            net = conv(net, 4 * base_channel, 3, 1, normalizer_fn, activation_fn, 4)

            net = self.residual_block(net, 4 * base_channel, normalizer_fn, activation_fn, 5)
            net = self.residual_block(net, 4 * base_channel, normalizer_fn, activation_fn, 7)
            net = self.residual_block(net, 4 * base_channel, normalizer_fn, activation_fn, 9)
            net = self.residual_block(net, 4 * base_channel, normalizer_fn, activation_fn, 11)

            net = self.decode_img(net, base_channel, net_ss, normalizer_fn, activation_fn, 13)

            # 64, 3
            generated_img = net
            return generated_img

    def encode_pic(self, image_input, base_channel, normalizer_fn, activation_fn):
        # s = 64
        net_s = image_input
        net_s2 = conv(image_input, base_channel, 4, 2, None, activation_fn, 0)
        net_s4 = conv(net_s2, 2 * base_channel, 4, 2, normalizer_fn, activation_fn, 1)
        net_s8 = conv(net_s4, 4 * base_channel, 4, 2, normalizer_fn, activation_fn, 2)
        # To obtain a better generated fake pic, just down-sample the segmentation to s8

        return net_s8, [net_s, net_s2, net_s4]

    def wrap_emb(self, emb, tile_size):
        return tile(emb, tile_size)

    def residual_block(self, input_, channel, normalizer_fn, activation_fn, start_index):
        input_tmp = input_
        net = conv(input_, channel, 3, 1, normalizer_fn, activation_fn, start_index)
        net = conv(net, channel, 3, 1, normalizer_fn, None, start_index + 1)
        return activation_fn(input_tmp + net)

    def decode_img(self, input_, base_channel, net_ss, normalizer_fn, activation_fn, start_index):
        [net_s, net_s2, net_s4] = net_ss
        # with tf.variable_scope('decode_pic'):
        net = upsample_conv_with_concat(input_, net_s4, base_channel, 3, 1, normalizer_fn, activation_fn, start_index)
        net = conv(net, base_channel, 3, 1, normalizer_fn, activation_fn, start_index + 1)
        net = conv(net, base_channel, 3, 1, normalizer_fn, activation_fn, start_index + 2)
        net = upsample_conv_with_concat(net, net_s2, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 3)
        net = conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 4)
        net = conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 5)
        net = upsample_conv_with_concat(net, net_s, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 6)
        net = conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 7)
        net = conv(net, 3, 4, 1, None, tf.tanh, start_index + 8)

        # return net / 2 + 0.5
        return net

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

        # return net / 2 + 0.5


if __name__ == '__main__':
    emb = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1024])
    cap = tf.placeholder(tf.string, shape=[BATCH_SIZE, 1])
    seg = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 64, 64, N_CAT])
    g = Generator(emb, cap, 'g_img', seg=seg)
    print()
