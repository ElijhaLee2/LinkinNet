import tensorflow as tf
from model_911.Generator import Generator
from model_911.Discriminator import Discriminator
from model_911.Optimizer_1 import GeneratorOptimizer, DiscriminatorOptimizer
from other.config import BATCH_SIZE, IS_STACK_0, IS_STACK_1, CAT_NUMs

"""
add gp_rw
process the input of D
"""


def build_whole_graph(batch):
    [img, img_w, seg, seg_w, embedding, caption] = batch

    models = dict()

    # Stack 0 ---------------------------------------------------
    # if IS_STACK_0:
    #     #   gen_seg -----------
    #     generator_seg = Generator(embedding, caption, 'gen_seg')
    #     #   dis_seg -----------
    #     discriminator_seg_real = Discriminator(seg, embedding, 'dis_seg', 'real')
    #     discriminator_seg_fake = Discriminator(generator_seg.generated_pic, embedding, 'dis_seg', 'fake', True)
    #     discriminator_seg_wrong = Discriminator(seg, wrong_embedding, 'dis_seg', 'wrong', True)
    #     interpolates_seg = seg + \
    #                        (tf.random_uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.) * (
    #                            generator_seg.generated_pic - seg))
    #     discriminator_seg_penalty = Discriminator(interpolates_seg, embedding, 'dis_seg', 'penalty', True)
    #
    #     models['STACK_0'] = {
    #         'gen': generator_seg,
    #         'dis_real': discriminator_seg_real,
    #         'dis_fake': discriminator_seg_fake,
    #         'dis_wrong': discriminator_seg_wrong,
    #         'dis_penalty': discriminator_seg_penalty, }

    # Stack 1 ---------------------------------------------------
    if IS_STACK_1:
        #   gen_img -----------
        generator_img = Generator(embedding, 'gen_img', seg=seg)
        img_r_ = mask_img(img, seg, CAT_NUMs)
        # TODO one real image for D
        t = tf.transpose(tf.reshape(img_r_[0], [64, 64, 3, 3]), [2, 0, 1, 3])
        tf.summary.image('img_masked', t[2], max_outputs=1)

        img_f_ = mask_img(generator_img.generated_pic, seg, CAT_NUMs)
        img_w_ = mask_img(img_w, seg_w, CAT_NUMs)
        #   dis_img -----------
        discriminator_img_real = Discriminator(img_r_, seg, embedding, 'dis_img', 'real')
        discriminator_img_fake = Discriminator(img_f_, seg, embedding, 'dis_img', 'fake', True)
        discriminator_img_wrong = Discriminator(img_w_, seg_w, embedding, 'dis_img', 'wrong', True)

        inter_rf_img = img_r_ + (
            tf.random_uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.) * (img_f_ - img_r_))
        inter_rw_img = img_r_ + (
            tf.random_uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.) * (img_w_ - img_r_))
        inter_rw_seg = seg + (
            tf.random_uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.) * (seg_w - seg))
        discriminator_gp_rf = Discriminator(inter_rf_img, seg, embedding, 'dis_img', 'gp_rf', True)
        discriminator_gp_rw = Discriminator(inter_rw_img, inter_rw_seg, embedding, 'dis_img', 'gp_rw', True)

        models['STACK_1'] = {
            'gen': generator_img,
            'dis_real': discriminator_img_real,
            'dis_fake': discriminator_img_fake,
            'dis_wrong': discriminator_img_wrong,
            'dis_gp_rf': discriminator_gp_rf,
            'dis_gp_rw': discriminator_gp_rw, }

    # Optimizer
    optmzrs = dict()

    # if IS_STACK_0:
    #     optmzr_dis_seg = DiscriminatorOptimizer(models['STACK_0'], 'optmzr_dis_seg')
    #     optmzr_gen_seg = GeneratorOptimizer(models['STACK_0'], 'optmzr_gen_seg')
    #     optmzrs['STACK_0'] = {
    #         'optmzr_dis': optmzr_dis_seg,
    #         'optmzr_gen': optmzr_gen_seg, }

    if IS_STACK_1:
        optmzr_dis_img = DiscriminatorOptimizer(models['STACK_1'], 'optmzr_dis_img')
        optmzr_gen_img = GeneratorOptimizer(models['STACK_1'], 'optmzr_gen_img')
        optmzrs['STACK_1'] = {
            'optmzr_dis': optmzr_dis_img,
            'optmzr_gen': optmzr_gen_img, }

    return models, optmzrs


def mask_img(img, seg, class_no):
    img_ = tf.tile(tf.expand_dims(img, 3), [1, 1, 1, len(class_no), 1])  # [batch_size,h,w,class,3]
    seg_ = tf.tile(tf.expand_dims(seg, 4), [1, 1, 1, 1, 3])  # [batch_size,h,w,class,3]
    masked = img_ * seg_  # shape should be BS * IMG_SIZE * IMG_SIZE * class * 3
    masked_shape = masked.get_shape().as_list()
    masked_img = tf.reshape(masked, masked_shape[0:3] + [masked_shape[-2] * masked_shape[-1]])

    # masked_img_channel_first = tf.transpose(masked_img, [3, 0, 1, 2])
    # channel_no = [0, 1, 2]
    # for c in class_no:
    #     channel_no += [c * 3, c * 3 + 1, c * 3 + 2]
    # masked_img_ = tf.gather(masked_img_channel_first, channel_no)
    # masked_img_ = tf.transpose(masked_img_, [1, 2, 3, 0])

    return tf.concat([img, masked_img], axis=-1)  # concat the real image together

# def sum_one_mask_img(image_input):
#     img = image_input[0]
#     shape = img.get_shape().as_list()
#     assert shape[-1] == (N_CAT + 1) * 3
#     img_ = tf.reshape(img, shape[0:2] + [3, shape[2] // 3])
#     img_ = tf.transpose(img_, [3, 0, 1, 2])
#     summarize(TB_GROUP.outputs, 'masked_img', img_, 'img')
# if __name__ == '__main__':
# [img, seg, embedding, wrong_embedding, caption] \
#     = [tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3]),
#        tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 91]),
#        tf.placeholder(tf.float32, [BATCH_SIZE, 1024]),
#        tf.placeholder(tf.float32, [BATCH_SIZE, 1024]),
#        tf.placeholder(tf.string, [BATCH_SIZE, 1])]
# build_whole_graph([img, seg, embedding, wrong_embedding, caption])
