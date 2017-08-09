import tensorflow as tf
import tensorflow.contrib.layers as ly
from other.hyperparameter import GENERATOR_HP
from other.config import BATCH_SIZE, N_CLASS


class Generator:
    def __init__(self, embedding, caption, name: str, seg=None, reuse=False):
        with tf.name_scope(name) as scope:
            if name.find('seg') != -1:
                assert seg is None
                self.generated_pic = self._build(embedding, name, reuse)
            elif name.find('img') != -1:
                assert type(seg) == tf.Tensor
                self.generated_pic = self._build_2(seg, embedding, name, reuse)
            else:
                raise ValueError('Nether \'seg\' nor \'img\' is in \'name\' of Generator')

            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            self.model_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=name)

            # self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,scope=scope))
            tf.summary.text('caption', caption)
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)

    def _build(self, embedding, name, reuse):
        z_dim = GENERATOR_HP['z_dim']
        base_channel = GENERATOR_HP['base_channel']
        kernel_size = GENERATOR_HP['kernel_size']
        stride = GENERATOR_HP['stride']
        activation_fn = GENERATOR_HP['activation_fn']
        normalizer_fn = GENERATOR_HP['normalizer_fn']
        weights_initializer = GENERATOR_HP['weights_initializer']
        with tf.variable_scope(name, reuse=reuse) as scope:
            z = tf.random_normal([BATCH_SIZE, z_dim])
            input = tf.concat([z, embedding], axis=1)
            net = ly.fully_connected(input, 2 * 2 * base_channel * 16, activation_fn, normalizer_fn=None,
                                     weights_initializer=weights_initializer, scope='z_emb_fc')
            net = tf.reshape(net, [BATCH_SIZE, 2, 2, base_channel * 16])
            # 2, 1024
            net = ly.conv2d_transpose(net, base_channel * 8, kernel_size, stride,
                                      activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                                      weights_initializer=weights_initializer, scope='0_deconv')
            # 4, 512
            net = ly.conv2d_transpose(net, base_channel * 4, kernel_size, stride,
                                      activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                                      weights_initializer=weights_initializer, scope='1_deconv')
            # 8, 256
            net = ly.conv2d_transpose(net, base_channel * 2, kernel_size, stride,
                                      activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                                      weights_initializer=weights_initializer, scope='2_deconv')
            # 16, 128
            net = ly.conv2d_transpose(net, base_channel * 1, kernel_size, stride,
                                      activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                                      weights_initializer=weights_initializer, scope='3_deconv')
            # 32, 64

            #  segmentation
            net = ly.conv2d_transpose(net, N_CLASS, kernel_size, stride,
                                      activation_fn=activation_fn, normalizer_fn=None,
                                      weights_initializer=weights_initializer, scope='4_net2_deconv')
            # 64, N_CLASS
            net = tf.nn.softmax(net)

            return net

    def _build_2(self, seg, emb, name, reuse):
        base_channel = GENERATOR_HP['base_channel']
        kernel_size = GENERATOR_HP['kernel_size']
        stride = GENERATOR_HP['stride']
        activation_fn = GENERATOR_HP['activation_fn']
        normalizer_fn = GENERATOR_HP['normalizer_fn']
        weights_initializer = GENERATOR_HP['weights_initializer']

        with tf.variable_scope(name, reuse=reuse) as scope:
            # 64, N_CLASS
            net1 = ly.conv2d(seg, base_channel, kernel_size, stride,
                             activation_fn=activation_fn, normalizer_fn=None, scope='0_conv',
                             weights_initializer=weights_initializer)
            # 32, 64
            net2 = ly.conv2d(net1, base_channel * 2, kernel_size, stride,
                             activation_fn=activation_fn, normalizer_fn=normalizer_fn, scope='1_conv',
                             weights_initializer=weights_initializer)
            # 16, 128
            net3 = ly.conv2d(net2, base_channel * 4, kernel_size, stride,
                             activation_fn=activation_fn, normalizer_fn=normalizer_fn, scope='2_conv',
                             weights_initializer=weights_initializer)
            # 8, 256
            net4 = ly.conv2d(net3, base_channel * 8, kernel_size, stride,
                             activation_fn=activation_fn, normalizer_fn=normalizer_fn, scope='3_conv',
                             weights_initializer=weights_initializer)
            # 4, 512

            # Embedding: 4,256
            reduced_emb = ly.fully_connected(emb, 256, activation_fn=activation_fn, normalizer_fn=None,
                                             weights_initializer=weights_initializer, scope='emb_fc')
            reduced_emb = tf.expand_dims(reduced_emb, 1)
            reduced_emb = tf.expand_dims(reduced_emb, 2)
            reduced_emb = tf.tile(reduced_emb, [1, 4, 4, 1])

            # 4, 512+256, Concat emb
            concat = tf.concat([net4, reduced_emb], axis=3)
            net5 = ly.conv2d_transpose(concat, base_channel * 8, kernel_size, 1,
                                       activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                                       weights_initializer=weights_initializer, scope='0_deconv')
            # 4, 512+512
            net5_4 = tf.concat([net5, net4], axis=3)
            net6 = ly.conv2d_transpose(net5_4, base_channel * 4, kernel_size, stride,
                                       activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                                       weights_initializer=weights_initializer, scope='1_deconv')

            # 8, 256+256
            net6_3 = tf.concat([net6, net3], axis=3)
            net7 = ly.conv2d_transpose(net6_3, base_channel * 2, kernel_size, stride,
                                       activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                                       weights_initializer=weights_initializer, scope='2_deconv')
            # 16, 128+128
            net7_2 = tf.concat([net7, net2], axis=3)
            net8 = ly.conv2d_transpose(net7_2, base_channel * 1, kernel_size, stride,
                                       activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                                       weights_initializer=weights_initializer, scope='3_deconv')
            # 32, 64+64
            net8_1 = tf.concat([net8, net1], axis=3)
            net9 = ly.conv2d_transpose(net8_1, 3, kernel_size, stride,
                                       activation_fn=activation_fn, normalizer_fn=None,
                                       weights_initializer=weights_initializer, scope='4_deconv')
            # 64, 3
            generated_img = tf.tanh(net9) / 2 + 0.5

            return generated_img


if __name__ == '__main__':
    emb = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1024])
    cap = tf.placeholder(tf.string, shape=[BATCH_SIZE, 1])
    seg = tf.placeholder(tf.float32, shape=[BATCH_SIZE,64,64,N_CLASS])
    g = Generator(emb, cap, 'g_img', seg=seg)
    print()
