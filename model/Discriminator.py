import tensorflow as tf

from model.hyperparameter import DISCRIMINATOR_HP
from other.function import tile, l2_norm_square
from other.nn_func import conv, dense


class Discriminator:
    def __init__(self, image_input, text_embedding, variable_scope_name: str, additional_name: str, reuse=False):
        self.variable_scope_name = variable_scope_name
        full_name = variable_scope_name + '_' + additional_name
        with tf.name_scope(full_name) as scope:
            # image_input = tf.identity(image_input)
            self.image_input = tf.identity(image_input)
            self.emb_input = tf.identity(text_embedding)  # FUUUUUUUUUUUUUCK!!!!!!!!!
            self.score = self._build(self.image_input, self.emb_input, variable_scope_name, reuse)
            self.trainable_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variable_scope_name)
            if additional_name.find('gp') == -1:
                tf.summary.scalar('score', tf.reduce_mean(self.score))

            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)

    def _build(self, image_input, text_embedding, variable_scope_name, reuse):
        base_channel = DISCRIMINATOR_HP['base_channel']
        # kernel_size = DISCRIMINATOR_HP['kernel_size']
        # stride = DISCRIMINATOR_HP['stride']
        activation_fn = DISCRIMINATOR_HP['activation_fn']
        normalizer_fn = DISCRIMINATOR_HP['normalizer_fn']
        emb_reduced_dim = DISCRIMINATOR_HP['emb_reduced_dim']
        # weights_initializer = DISCRIMINATOR_HP['weights_initializer']
        # biases_initializer = DISCRIMINATOR_HP['biases_initializer']

        with tf.variable_scope(variable_scope_name, reuse=reuse):
            encoded_pic = self.encode_pic(image_input, base_channel, normalizer_fn, activation_fn)
            shape = encoded_pic.get_shape().as_list()
            reduced_emb = self.wrap_emb(text_embedding, emb_reduced_dim, shape[1], activation_fn)

            # Concat
            concat = tf.concat([encoded_pic, reduced_emb], axis=3)

            # ATTENTION: ly_index !
            net = conv(concat, 8 * base_channel, 1, 1, normalizer_fn, activation_fn, 7)

            gi,ge = tf.gradients(net,[encoded_pic,reduced_emb])
            # tf.summary.histogram('encoded_pic',encoded_pic)
            # tf.summary.histogram('reduced_emb',reduced_emb)

            tf.summary.scalar('grad_con_img_sclr',tf.reduce_mean(tf.sqrt(l2_norm_square(gi))))
            tf.summary.histogram('grad_con_img_his',gi)
            tf.summary.scalar('grad_con_emb_sclr',tf.reduce_mean(tf.sqrt(l2_norm_square(ge))))
            tf.summary.histogram('grad_con_emb_his',ge)


            # net = conv(net, 4 * base_channel, 2, 1, normalizer_fn, activation_fn, 8)
            net = tf.reshape(net, [net.get_shape().as_list()[0], -1])
            score = dense(net, 1, None, None, 9)

        return score  # (batch_size,1)

    def encode_pic(self, image_input, base_channel, normalizer_fn, activation_fn):
        # s, bc --> s16, 8*bc
        # with tf.variable_scope('encode_pic'):
        # s = 64
        net_s2 = conv(image_input, base_channel, 4, 2, None, activation_fn, 0)
        net_s4 = conv(net_s2, 2 * base_channel, 4, 2, normalizer_fn, activation_fn, 1)
        net_s8 = conv(net_s4, 4 * base_channel, 4, 2, normalizer_fn, activation_fn, 2)
        net_s16_0 = conv(net_s8, 8 * base_channel, 4, 2, normalizer_fn, activation_fn, 3)
        net_s16 = conv(net_s16_0, 16 * base_channel, 1, 1, normalizer_fn, activation_fn, 4)
        net_s16 = conv(net_s16, 8 * base_channel, 1, 1, normalizer_fn, activation_fn, 5)
        net_s16 = conv(net_s16, 8 * base_channel, 3, 1, normalizer_fn, None, 6)
        # residual
        # net_s16 = activation_fn(net_s16 + net_s16_0)
        net_s16 = activation_fn(net_s16)

        return net_s16

    def wrap_emb(self, emb, emb_reduced_dim, tile_size, activation_fn):
        reduced_emb = dense(emb, emb_reduced_dim, None, activation_fn, '0_emb_fc')

        # reduced_emb = tf.expand_dims(reduced_emb, 1)
        # reduced_emb = tf.expand_dims(reduced_emb, 2)
        # reduced_emb = tf.tile(tf.expand_dims(tf.expand_dims(reduced_emb, 1), 2), [1, tile_size, tile_size, 1],
        #                       name='tile_emb')
        # a = tf.expand_dims(reduced_emb, 1)
        # a = tf.tile(a, [1, tile_size, 1])
        # a = tf.expand_dims(a, 1)
        # reduced_emb = tf.tile(a, [1, tile_size, 1, 1])
        reduced_emb = tile(reduced_emb, [tile_size, tile_size])
        return reduced_emb



# TEST
if __name__ == '__main__':
    img = tf.placeholder(tf.float32, [32, 64, 64, 3])
    emb = tf.placeholder(tf.float32, [32, 1024])
    g = Discriminator(img, emb, 'd', 'real')
