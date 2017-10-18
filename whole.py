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


def _imgbuild_whole_graph(batch):
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
        img_f_ = generator_img.generated_pic

        tf.summary.image('gen', img_f_, max_outputs=BATCH_SIZE, collections=['LinkNet_sum'])

        models['STACK_1'] = {
            'gen': generator_img}

    # Optimizer
    optmzrs = dict()

    if IS_STACK_1:
        optmzr_dis_img = None
        optmzr_gen_img = None
        optmzrs['STACK_1'] = {
            'optmzr_dis': optmzr_dis_img,
            'optmzr_gen': optmzr_gen_img, }

    return models, optmzrs
