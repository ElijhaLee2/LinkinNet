import tensorflow as tf
import tensorflow.contrib.layers as ly
from other.hyperparameter import GENERATOR_HP
from other.config import BATCH_SIZE, N_CLASS


class Generator():
    def __init__(self, embedding, caption, name, reuse=False):
        with tf.name_scope(name) as scope:
            self.generated_pic = self._build(embedding, name, reuse)
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            self.model_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=name)

            # self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,scope=scope))
            tf.summary.text('caption', caption)
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)
        pass

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
            net = ly.fully_connected(input, 4 * 4 * base_channel * 16, activation_fn, normalizer_fn=None,
                                     weights_initializer=weights_initializer, scope='z_emb_fc')
            net = tf.reshape(net, [BATCH_SIZE, 4, 4, base_channel * 16])
            # 4, 512
            net = ly.conv2d_transpose(net, base_channel * 8, kernel_size, stride,
                                      activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                                      weights_initializer=weights_initializer, scope='0_deconv')
            # 8, 256
            net = ly.conv2d_transpose(net, base_channel * 4, kernel_size, stride,
                                      activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                                      weights_initializer=weights_initializer, scope='1_deconv')
            # 16, 128
            net = ly.conv2d_transpose(net, base_channel * 2, kernel_size, stride,
                                      activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                                      weights_initializer=weights_initializer, scope='2_deconv')
            # 32, 64

            #  segmentation
            # 32, 64
            net = ly.conv2d_transpose(net, N_CLASS, kernel_size, stride,
                                       activation_fn=activation_fn, normalizer_fn=None,
                                       weights_initializer=weights_initializer, scope='3_net2_deconv')
            # 64, N_CLASS
            # net2 = ly.conv2d_transpose(net2, N_CLASS, kernel_size, stride,
            #                           activation_fn=None, normalizer_fn=None,
            #                           weights_initializer=weights_initializer,  scope='4_net2_deconv')
            # # 128, N_CLASS
            net = tf.nn.softmax(net)

            return net
