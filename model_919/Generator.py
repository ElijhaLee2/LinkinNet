import tensorflow as tf

from model_919.hyperparameter import GENERATOR_HP
from other.config import BATCH_SIZE, N_CAT
from other.nn_func import conv, upsample_conv_with_concat, upsample_conv, dense, deconv
from model_919.model_func import residual_block


class Generator:
    def __init__(self, embedding, name: str, batch_size=BATCH_SIZE, seg=None, reuse=False):
        with tf.name_scope(name) as scope:
            self.embedding = tf.identity(embedding)
            if name.find('seg') != -1:
                assert seg is None
                self.generated_pic = self._build_0(self.embedding, name, reuse, batch_size=batch_size)
            elif name.find('img') != -1:
                assert type(seg) == tf.Tensor
                self.seg = tf.identity(seg)
                self.generated_pic = self._build_1(self.seg, self.embedding, name, reuse, batch_size=batch_size)
            else:
                raise ValueError('Nether \'seg\' nor \'img\' is in \'name\' of Generator')

            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            self.model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)

    def _build_0(self, embedding, name, reuse, batch_size=BATCH_SIZE):
        z_dim = GENERATOR_HP['z_dim']
        gf_dim = GENERATOR_HP['gf_dim']
        activation_fn = GENERATOR_HP['activation_fn']
        normalizer_fn = GENERATOR_HP['normalizer_fn']

        with tf.variable_scope(name, reuse=reuse) as scope:
            z = tf.random_normal([batch_size, z_dim])
            input_ = tf.concat([z, embedding], axis=1)
            net = dense(input_, 4 * 4 * 8 * gf_dim, None, None, 0)
            net = tf.reshape(net, [batch_size, 4, 4, 8 * gf_dim])
            # 4,8*gf_dim
            net = residual_block(net, 4 * gf_dim, normalizer_fn, activation_fn, 1)
            # 4,8*gf_dim

            net = residual_block(net, 4 * gf_dim, normalizer_fn, activation_fn, 3)
            net = residual_block(net, 4 * gf_dim, normalizer_fn, activation_fn, 5)
            net = residual_block(net, 4 * gf_dim, normalizer_fn, activation_fn, 7)

            # 64, 91
            generated_img = self.decode_seg(net, gf_dim, normalizer_fn, activation_fn, 9)
        return generated_img

    def _build_1(self, seg, emb, name, reuse, batch_size=BATCH_SIZE):
        gf_dim = GENERATOR_HP['gf_dim']
        act_fn = GENERATOR_HP['act_fn']
        norm_fn = GENERATOR_HP['norm_fn']
        norm_param = GENERATOR_HP['norm_params']
        z_dim = GENERATOR_HP['z_dim']

        with tf.variable_scope(name, reuse=reuse) as scope:
            # emb
            z = tf.random_normal([batch_size, z_dim], name='z')
            emb_z = tf.concat([emb, z], axis=-1, name='emb_z_concat')
            emb_net = dense(emb_z, 4 * 4 * 8 * gf_dim, None, None, None, 0)
            emb_net = tf.reshape(emb_net, [batch_size, 4, 4, 8 * gf_dim])
            # 4, 8*gf
            emb_net = residual_block(emb_net, 8 * gf_dim, norm_fn, norm_param, act_fn, 1)
            emb_net = act_fn(emb_net)
            # 4, 8*gf

            # seg
            net1, net2, net4, net8, net16 = self.encode_seg(self.seg, gf_dim, norm_fn, norm_param, act_fn, 0)

            # concat
            net = tf.concat([emb_net, net16], axis=-1)
            # 4,16*gf
            net = conv(net, 8 * gf_dim, 3, 1, norm_fn, norm_param, None, 4)
            # 4, 8*gf
            net = residual_block(net, 8 * gf_dim, norm_fn, norm_param, act_fn, 5)
            net = residual_block(net, 8 * gf_dim, norm_fn, norm_param, act_fn, 8)
            net = residual_block(net, 8 * gf_dim, norm_fn, norm_param, act_fn, 11)
            net = act_fn(net)

            # decode
            # 4,8*gf
            net = upsample_conv_with_concat(net, net8, 4 * gf_dim, 4, 1, norm_fn, norm_param, act_fn, 14)
            # 8,4*gf
            net = upsample_conv_with_concat(net, net4, 2 * gf_dim, 4, 1, norm_fn, norm_param, act_fn, 15)
            # 16,2*gf
            net = upsample_conv_with_concat(net, net2, gf_dim, 4, 1, norm_fn, norm_param, act_fn, 16)
            # 32,gf
            net = upsample_conv_with_concat(net, net1, gf_dim / 2, 4, 1, norm_fn, norm_param, act_fn, 17)
            # 64,gf/2
            net = conv(net, 3, 4, 1, None, None, tf.tanh, 18)

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



    # def upsample(self, feature_map, label, norm_fn, ly_index):
    #     [_, img_length, _, c] = feature_map.get_shape().as_list()
    #     fm_up = tf.image.resize_bilinear(feature_map, [img_length * 2, img_length * 2])
    #     concat = tf.concat([fm_up, label], axis=-1)
    #     # ATTENTION: res is NOT ACT!
    #     res = conv(concat, c // 2, 3, 1, norm_fn, None, ly_index)
    #     return res

    def encode_seg(self, image_input, gf_dim, norm_fn, norm_param, act_fn, start_index):
        # 4 layers
        with tf.variable_scope('encode_seg'):
            net = image_input
            # 64,3*3
            net2 = conv(net, gf_dim, 5, 2, None, norm_param, act_fn, start_index)
            # 32,df
            net4 = conv(net2, gf_dim * 2, 4, 2, norm_fn, norm_param, act_fn, start_index + 1)
            # 16,2*df
            net8 = conv(net4, gf_dim * 4, 4, 2, norm_fn, norm_param, act_fn, start_index + 2)
            # 8,4*df
            net16 = conv(net8, gf_dim * 8, 3, 2, norm_fn, norm_param, act_fn, start_index + 3)
            # 4,8*df
        return net, net2, net4, net8, net16

        # def decode_seg(self, input_, base_channel, normalizer_fn, activation_fn, start_index):
        #     net = upsample_conv(input_, base_channel, 3, 1, normalizer_fn, activation_fn, start_index)
        #     net = conv(net, base_channel, 3, 1, normalizer_fn, activation_fn, start_index + 1)
        #     net = conv(net, base_channel, 3, 1, normalizer_fn, activation_fn, start_index + 2)
        #     net = upsample_conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 3)
        #     net = conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 4)
        #     net = conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 5)
        #     net = upsample_conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 6)
        #     net = conv(net, base_channel, 4, 1, normalizer_fn, activation_fn, start_index + 7)
        #     net = conv(net, N_CAT, 4, 1, None, None, start_index + 8)
        #
        #     return net / 2 + 0.5


if __name__ == '__main__':
    emb = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1024])
    # cap = tf.placeholder(tf.string, shape=[BATCH_SIZE, 1])
    seg = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 64, 64, N_CAT])
    g = Generator(emb, 'g_img', seg=seg)
    print()
