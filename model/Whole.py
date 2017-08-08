import tensorflow as tf
from model.Generator import Generator
from model.Discriminator import Discriminator
from model.Optimizer import GeneratorOptimizer, DiscriminatorOptimizer, MatchNetOptimizer
from other.config import MATCHNET_IMG_NAME, MATCHNET_SEG_NAME, GENERATOR_NAME, DISCRIMINATOR_IMG_NAME, \
    DISCRIMINATOR_SEG_NAME, OPTIMIZER_DIS_IMG_NAME, OPTIMIZER_DIS_SEG_NAME, OPTIMIZER_GEN_NAME, BATCH_SIZE


def build_whole_graph(batch):
    [img, seg, embedding, wrong_embedding, caption] = batch
    # Stack I -----------
    #   Generator ---------------------------------------------------
    generator = Generator(embedding, caption, 'Generator')
    #   dis_Seg
    discriminator_seg_real = Discriminator(seg, embedding, DISCRIMINATOR_SEG_NAME, 'real')
    discriminator_seg_fake = Discriminator(generator.generated_pic, embedding, DISCRIMINATOR_SEG_NAME, 'fake', True)
    discriminator_seg_wrong = Discriminator(seg, wrong_embedding, DISCRIMINATOR_SEG_NAME, 'wrong', True)
    interpolates_seg = seg + \
                       (tf.random_uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.) * (generator.generated_pic - seg))
    discriminator_seg_penalty = Discriminator(interpolates_seg, embedding, DISCRIMINATOR_SEG_NAME, 'penalty', True)


    # Pack models
    models = {
        GENERATOR_NAME: generator,
        DISCRIMINATOR_SEG_NAME + '_real': discriminator_seg_real,
        DISCRIMINATOR_SEG_NAME + '_fake': discriminator_seg_fake,
        DISCRIMINATOR_SEG_NAME + '_wrong': discriminator_seg_wrong,
        DISCRIMINATOR_SEG_NAME + '_penalty': discriminator_seg_penalty,
    }

    # Optimizer
    optmzr_dis_seg = DiscriminatorOptimizer(models, DISCRIMINATOR_SEG_NAME, OPTIMIZER_DIS_SEG_NAME)
    optmzr_gen = GeneratorOptimizer(models, OPTIMIZER_GEN_NAME)

    # Pack optimizers
    optmzrs = {
        OPTIMIZER_DIS_SEG_NAME: optmzr_dis_seg,
        OPTIMIZER_GEN_NAME: optmzr_gen,
    }

    return models, optmzrs
