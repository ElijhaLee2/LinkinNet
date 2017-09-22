import tensorflow as tf

from model_919.hyperparameter import DISCRIMINATOR_HP
from other.config import N_CAT, TB_GROUP, SUM_COLLEC
from other.function import tile, summarize
from other.nn_func import conv, dense
from model_919.model_func import residual_block


class Discriminator:
    def __init__(self, image_input, seg_input, text_embedding, variable_scope_name: str, additional_name: str,
                 reuse=False):
        self.variable_scope_name = variable_scope_name
        full_name = variable_scope_name + '_' + additional_name
        with tf.name_scope(full_name) as scope:
            # image_input = tf.identity(image_input)
            self.image_input = tf.identity(image_input)
            self.seg_input = tf.identity(seg_input)
            self.emb_input = tf.identity(text_embedding)  # FUUUUUUUUUUUUUCK!!!!!!!!!
            self.score = self._build_1(self.image_input, self.seg_input, self.emb_input,
                                       variable_scope_name, additional_name, reuse)
            self.trainable_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variable_scope_name)
            if additional_name.find('gp') == -1:
                summarize(TB_GROUP.scores, additional_name, tf.reduce_mean(self.score), 'scl')

                # self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)

    def _build_1(self, image_input, seg_input, text_embedding, variable_scope_name, additional_name, reuse):
        df_dim = DISCRIMINATOR_HP['df_dim']
        act_fn = DISCRIMINATOR_HP['act_fn']
        norm_fn = DISCRIMINATOR_HP['norm_fn']
        emb_reduced_dim = DISCRIMINATOR_HP['emb_reduced_dim']

        with tf.variable_scope(variable_scope_name, reuse=reuse):
            encoded_img = self.encode_img(image_input, df_dim, norm_fn, act_fn, 0)
            encoded_seg = self.encode_seg(seg_input, df_dim, norm_fn, act_fn, 0)
            shape = encoded_img.get_shape().as_list()
            reduced_emb = self.wrap_emb(text_embedding, emb_reduced_dim, shape[1], act_fn)

            # summarize('other_out', '%s/encoded_img' % additional_name, encoded_img, 'his')
            # summarize('other_out', '%s/reduced_emb' % additional_name, reduced_emb, 'his')
            summarize('other_out', '%s/encoded_img' % additional_name, tf.global_norm([encoded_img]), 'scl')
            summarize('other_out', '%s/encoded_seg' % additional_name, tf.global_norm([encoded_seg]), 'scl')
            summarize('other_out', '%s/reduced_emb' % additional_name, tf.global_norm([reduced_emb]), 'scl')

            # Concat
            concat = tf.concat([encoded_img, encoded_seg, reduced_emb], axis=3)
            # 4, 24*df
            # ATTENTION: ly_index !
            net = conv(concat, df_dim * 8, 1, 1, norm_fn, act_fn, 7)
            # 4, 8*df
            score = conv(net, 1, 4, 1, None, None, 8, padding='VALID')  # 1,1 ,equal to fc

        return score  # (batch_size,)

    def encode_img(self, image_input, df_dim, norm_fn, act_fn, start_index):
        with tf.variable_scope('encode_img'):
            # n_cat = image_input.get_shape().as_list()[3] / 3
            # net = conv(image_input, n_cat, 1, 1, None, act_fn, start_index)
            net = image_input
            # 64,3*3
            net = conv(net, df_dim, 5, 2, None, act_fn, start_index)
            # 32,df
            net = conv(net, df_dim * 2, 4, 2, norm_fn, act_fn, start_index + 1)
            # 16,2*df
            net = conv(net, df_dim * 4, 4, 2, norm_fn, act_fn, start_index + 2)
            # 8,4*df
            net = conv(net, df_dim * 8, 4, 2, norm_fn, None, start_index + 3)
            # 4,8*df
            net = residual_block(net, df_dim * 8, norm_fn, act_fn, start_index + 4)
            net = act_fn(net)
            # 4,8*df=512
        return net

    def encode_seg(self, image_input, df_dim, norm_fn, act_fn, start_index):
        with tf.variable_scope('encode_seg'):
            net = image_input
            # 64,3*3  act_fn=None, cuz residual_block will act first
            net = conv(net, df_dim, 5, 2, None, act_fn, start_index)
            # 32,df
            net = conv(net, df_dim * 2, 4, 2, norm_fn, act_fn, start_index + 1)
            # 16,2*df
            net = conv(net, df_dim * 4, 4, 2, norm_fn, act_fn, start_index + 2)
            # 8,4*df
            net = conv(net, df_dim * 8, 4, 2, norm_fn, act_fn, start_index + 3)
            # 4,8*df
            # net = tf.random_normal([32, 4, 4, 8 * df_dim]) * image_input[:, 0:4, 0:4, 0:1]
        return net

    def wrap_emb(self, emb, emb_reduced_dim, tile_length, activation_fn):
        with tf.variable_scope('wrap_emb'):
            reduced_emb = dense(emb, emb_reduced_dim, None, activation_fn, '0_emb_fc')
            reduced_emb = tile(reduced_emb, [tile_length, tile_length])
        return reduced_emb


# TEST
if __name__ == '__main__':
    img = tf.placeholder(tf.float32, [32, 64, 64, 3])
    emb = tf.placeholder(tf.float32, [32, 1024])
    g = Discriminator(img, emb, 'd', 'real')
