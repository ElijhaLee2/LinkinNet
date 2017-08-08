import tensorflow as tf
import tensorflow.contrib.layers as ly
from other.config import BATCH_SIZE
from other.hyperparameter import DISCRIMINATOR_HP


class Discriminator():
    def __init__(self, image_input, text_embedding, variable_scope_name, additional_name: str, reuse=False):
        full_name = variable_scope_name + '_' + additional_name
        with tf.name_scope(full_name) as scope:
            self.image_input = image_input
            self.score = self._build(image_input, text_embedding, variable_scope_name, reuse)
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=variable_scope_name)
            self.model_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=variable_scope_name)

            tf.summary.scalar('score', tf.reduce_mean(self.score))
            # Only img will be summarized
            if full_name == DISCRIMINATOR_IMG_NAME + '_real':
                [tf.summary.histogram(var.name, var) for var in self.trainable_variables]
                tf.summary.image('real_image', image_input, max_outputs=BATCH_SIZE)
                tf.summary.histogram('real_image', image_input)

            if full_name == DISCRIMINATOR_IMG_NAME + '_fake':
                tf.summary.image('fake_image', image_input, max_outputs=BATCH_SIZE)
                tf.summary.histogram('fake_image', image_input)

            # self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope))
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)

    def _build(self, image_input, text_embedding, variable_scope_name, reuse):
        base_channel = DISCRIMINATOR_HP['base_channel']
        kernel_size = DISCRIMINATOR_HP['kernel_size']
        stride = DISCRIMINATOR_HP['stride']
        activation_fn = DISCRIMINATOR_HP['activation_fn']
        normalizer_fn = DISCRIMINATOR_HP['normalizer_fn']
        weights_initializer = DISCRIMINATOR_HP['weights_initializer']
        biases_initializer = DISCRIMINATOR_HP['biases_initializer']

        with tf.variable_scope(variable_scope_name, reuse=reuse):
            # 64, 3/N_CLASS
            net = ly.conv2d(image_input, base_channel * 2, kernel_size, stride,
                            activation_fn=activation_fn, normalizer_fn=None, scope='img_0_conv',
                            weights_initializer=weights_initializer, biases_initializer=biases_initializer)
            # 32, 64
            net = ly.conv2d(net, base_channel * 4, kernel_size, stride,
                            activation_fn=activation_fn, normalizer_fn=normalizer_fn, scope='img_1_conv',
                            weights_initializer=weights_initializer, biases_initializer=biases_initializer)
            # 16, 128
            net = ly.conv2d(net, base_channel * 8, kernel_size, stride,
                            activation_fn=activation_fn, normalizer_fn=normalizer_fn, scope='img_2_conv',
                            weights_initializer=weights_initializer, biases_initializer=biases_initializer)
            # 8, 256

            # Embedding
            # 1, 1024
            expand_embedding = tf.expand_dims(text_embedding, 1)
            expand_embedding = tf.expand_dims(expand_embedding, 2)
            emb = ly.conv2d_transpose(expand_embedding, 256, 2, 2, activation_fn=activation_fn,
                                      normalizer_fn=normalizer_fn, scope='emb_0_deconv',
                                      weights_initializer=weights_initializer, biases_initializer=biases_initializer)
            # 2, 256
            emb = ly.conv2d_transpose(emb, 128, 2, 2, activation_fn=activation_fn,
                                      normalizer_fn=normalizer_fn, scope='emb_1_deconv',
                                      weights_initializer=weights_initializer, biases_initializer=biases_initializer)
            # 4, 128
            emb = ly.conv2d_transpose(emb, 64, 2, 2, activation_fn=activation_fn,
                                      normalizer_fn=normalizer_fn, scope='emb_2_deconv',
                                      weights_initializer=weights_initializer, biases_initializer=biases_initializer)
            # 8, 64

            # concat embedding
            concat = tf.concat([net, emb], axis=3)

            # 8, 64+256=320
            net = ly.conv2d(concat, base_channel * 16, kernel_size, stride,
                            activation_fn=activation_fn, normalizer_fn=normalizer_fn, scope='3_conv',
                            weights_initializer=weights_initializer, biases_initializer=biases_initializer)
            # 4, 512
            reshape = tf.reshape(net, [BATCH_SIZE, 4 * 4 * 512])
            # 4*4*512
            score = ly.fully_connected(reshape, 1,
                                       activation_fn=None, normalizer_fn=None, scope='4_fc',
                                       weights_initializer=weights_initializer, biases_initializer=biases_initializer)
            # 1

        return score  # (batch_size,1)
