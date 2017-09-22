import tensorflow as tf
from model.Generator import Generator
from model.Discriminator import Discriminator
from model.Optimizer import GeneratorOptimizer, DiscriminatorOptimizer
from other.config import BATCH_SIZE, IS_STACK_0, IS_STACK_1


def build_whole_graph(batch):
    [img, seg, embedding, wrong_embedding, caption] = batch

    models =dict()

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
        generator_img = Generator(embedding, caption, 'gen_img', seg=seg)
        #   dis_img -----------
        discriminator_img_real = Discriminator(img, embedding, 'dis_img', 'real')
        discriminator_img_fake = Discriminator(generator_img.generated_pic, embedding, 'dis_img', 'fake', True)
        discriminator_img_wrong = Discriminator(img, wrong_embedding, 'dis_img', 'wrong', True)
        interpolates_img = img + \
                           (tf.random_uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.) * (
                               generator_img.generated_pic - img))
        discriminator_img_penalty = Discriminator(interpolates_img, embedding, 'dis_img', 'penalty', True)

        models['STACK_1'] = {
            'gen': generator_img,
            'dis_real': discriminator_img_real,
            'dis_fake': discriminator_img_fake,
            'dis_wrong': discriminator_img_wrong,
            'dis_penalty': discriminator_img_penalty,}


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


if __name__ == '__main__':
    [img, seg, embedding, wrong_embedding, caption] \
        = [tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3]),
           tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 91]),
           tf.placeholder(tf.float32, [BATCH_SIZE, 1024]),
           tf.placeholder(tf.float32, [BATCH_SIZE, 1024]),
           tf.placeholder(tf.string, [BATCH_SIZE, 1])]
    build_whole_graph([img, seg, embedding, wrong_embedding, caption])
