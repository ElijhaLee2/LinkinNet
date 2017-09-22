import tensorflow as tf
from model_919.Generator import Generator
from model_919.Discriminator import Discriminator
from model_919.Optimizer import GeneratorOptimizer, DiscriminatorOptimizer
from model_919.vgg import VGG
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
        # G
        generator_img = Generator(embedding, 'gen_img', seg=seg)

        img_r_ = img
        img_f_ = generator_img.generated_pic
        img_w_ = img_w

        # D
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

        # VGG
        vgg_r = VGG(img_r_, seg=seg)
        vgg_f = VGG(img_f_, seg=None, reuse=True)

        models['STACK_1'] = {
            'gen': generator_img,
            'dis_real': discriminator_img_real,
            'dis_fake': discriminator_img_fake,
            'dis_wrong': discriminator_img_wrong,
            'dis_gp_rf': discriminator_gp_rf,
            'dis_gp_rw': discriminator_gp_rw,
            'vgg_real': vgg_r,
            'vgg_fake': vgg_f, }

    # Optimizer
    optmzrs = dict()

    if IS_STACK_1:
        optmzr_dis_img = DiscriminatorOptimizer(models['STACK_1'], 'optmzr_dis_img')
        optmzr_gen_img = GeneratorOptimizer(models['STACK_1'], 'optmzr_gen_img')
        optmzrs['STACK_1'] = {
            'optmzr_dis': optmzr_dis_img,
            'optmzr_gen': optmzr_gen_img, }

    return models, optmzrs
