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
            # self.model_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=variable_scope_name)

            tf.summary.scalar('score', tf.reduce_mean(self.score))
            # Only img will be summarized
            if additional_name == 'real':
                [tf.summary.histogram(var.name, var) for var in self.trainable_variables]
                tf.summary.image('real_image', image_input, max_outputs=BATCH_SIZE)
                tf.summary.histogram('real_image', image_input)

            if additional_name == 'fake':
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
            net = ly.conv2d(image_input, base_channel, kernel_size, stride,
                            activation_fn=activation_fn, normalizer_fn=None, scope='0_img_conv',
                            weights_initializer=weights_initializer, biases_initializer=biases_initializer)
            # 32, 32
            net = ly.conv2d(net, base_channel * 2, kernel_size, stride,
                            activation_fn=activation_fn, normalizer_fn=normalizer_fn, scope='1_img_conv',
                            weights_initializer=weights_initializer, biases_initializer=biases_initializer)
            # 16, 64
            net = ly.conv2d(net, base_channel * 4, kernel_size, stride,
                            activation_fn=activation_fn, normalizer_fn=normalizer_fn, scope='2_img_conv',
                            weights_initializer=weights_initializer, biases_initializer=biases_initializer)
            # 8, 128

            net = ly.conv2d(net, base_channel * 8, kernel_size, stride,
                            activation_fn=activation_fn, normalizer_fn=normalizer_fn, scope='3_img_conv',
                            weights_initializer=weights_initializer, biases_initializer=biases_initializer)
            # 4, 256

            # Embedding
            reduced_emb = ly.fully_connected(text_embedding, 256, activation_fn=activation_fn, normalizer_fn=None,
                                             weights_initializer=weights_initializer,
                                             biases_initializer=biases_initializer, scope='emb_fc')
            reduced_emb = tf.expand_dims(reduced_emb, 1)
            reduced_emb = tf.expand_dims(reduced_emb, 2)
            reduced_emb = tf.tile(reduced_emb, [1, 4, 4, 1])

            # Concat
            concat = tf.concat([net, reduced_emb], axis=3)

            # 4, 512
            net = ly.conv2d(net, base_channel * 16, kernel_size, stride,
                            activation_fn=activation_fn, normalizer_fn=normalizer_fn, scope='4_concat_conv',
                            weights_initializer=weights_initializer, biases_initializer=biases_initializer)

            # 2, 512
            reshape = tf.reshape(net, [BATCH_SIZE, 2 * 2 * 512])
            # 2*2*512
            score = ly.fully_connected(reshape, 1,
                                       activation_fn=None, normalizer_fn=None, scope='5_fc',
                                       weights_initializer=weights_initializer, biases_initializer=biases_initializer)
            # 1

        return score  # (batch_size,1)


# TEST
if __name__ == '__main__':
    img = tf.placeholder(tf.float32,[32,64,64,3])
    emb = tf.placeholder(tf.float32,[32,1024])
    g = Discriminator(img,emb,'d','real')