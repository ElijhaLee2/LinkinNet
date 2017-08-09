import tensorflow as tf
from model.Generator import Generator
from model.Discriminator import Discriminator
from model.Optimizer import GeneratorOptimizer, DiscriminatorOptimizer
from other.config import BATCH_SIZE


def build_whole_graph(batch):
    [img, seg, embedding, wrong_embedding, caption] = batch
    # Stack I -----------
    #   Generator ---------------------------------------------------
    generator = Generator(embedding, caption, 'gen_seg')
    #   dis_Seg
    discriminator_seg_real = Discriminator(seg, embedding, 'dis_seg', 'real')
    discriminator_seg_fake = Discriminator(generator.generated_pic, embedding, 'dis_seg', 'fake', True)
    discriminator_seg_wrong = Discriminator(seg, wrong_embedding, 'dis_seg', 'wrong', True)
    interpolates_seg = seg + \
                       (tf.random_uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.) * (
                           generator.generated_pic - seg))
    discriminator_seg_penalty = Discriminator(interpolates_seg, embedding, 'dis_seg', 'penalty', True)

    # Pack models
    models = {
        'STACK_0': {
            'gen': generator,
            'dis_real': discriminator_seg_real,
            'dis_fake': discriminator_seg_fake,
            'dis_wrong': discriminator_seg_wrong,
            'dis_penalty': discriminator_seg_penalty, }
    }

    # Optimizer
    optmzr_dis_seg = DiscriminatorOptimizer(models['STACK_0'], 'optmzr_dis_seg')
    optmzr_gen_seg = GeneratorOptimizer(models['STACK_0'], 'optmzr_gen_seg')

    # Pack optimizers
    optmzrs = {
        'STACK_0': {
            'optmzr_dis': optmzr_dis_seg,
            'optmzr_gen': optmzr_gen_seg, }
    }

    return models, optmzrs
