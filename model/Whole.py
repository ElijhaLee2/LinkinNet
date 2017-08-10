import tensorflow as tf
from model.Generator import Generator
from model.Discriminator import Discriminator
from model.Optimizer import GeneratorOptimizer, DiscriminatorOptimizer
from other.config import BATCH_SIZE


def build_whole_graph(batch):
    [img, seg, embedding, wrong_embedding, caption] = batch
    # Stack I ---------------------------------------------------
    #   gen_seg -----------
    generator_seg = Generator(embedding, caption, 'gen_seg')
    #   dis_seg -----------
    discriminator_seg_real = Discriminator(seg, embedding, 'dis_seg', 'real')
    discriminator_seg_fake = Discriminator(generator_seg.generated_pic, embedding, 'dis_seg', 'fake', True)
    discriminator_seg_wrong = Discriminator(seg, wrong_embedding, 'dis_seg', 'wrong', True)
    interpolates_seg = seg + \
                       (tf.random_uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.) * (
                           generator_seg.generated_pic - seg))
    discriminator_seg_penalty = Discriminator(interpolates_seg, embedding, 'dis_seg', 'penalty', True)

    # Stack I ---------------------------------------------------
    #   gen_seg -----------
    generator_img = Generator(embedding, caption, 'gen_img', seg=generator_seg.generated_pic)
    #   dis_seg -----------
    discriminator_img_real = Discriminator(img, embedding, 'dis_img', 'real')
    discriminator_img_fake = Discriminator(generator_img.generated_pic, embedding, 'dis_img', 'fake', True)
    discriminator_img_wrong = Discriminator(img, wrong_embedding, 'dis_img', 'wrong', True)
    interpolates_img = img + \
                       (tf.random_uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.) * (
                           generator_img.generated_pic - img))
    discriminator_img_penalty = Discriminator(interpolates_img, embedding, 'dis_img', 'penalty', True)

    # Pack models
    models = {
        'STACK_0': {
            'gen': generator_seg,
            'dis_real': discriminator_seg_real,
            'dis_fake': discriminator_seg_fake,
            'dis_wrong': discriminator_seg_wrong,
            'dis_penalty': discriminator_seg_penalty, },
        'STACK_1': {
            'gen': generator_img,
            'dis_real': discriminator_img_real,
            'dis_fake': discriminator_img_fake,
            'dis_wrong': discriminator_img_wrong,
            'dis_penalty': discriminator_img_penalty,
        }
    }

    # Optimizer
    optmzr_dis_seg = DiscriminatorOptimizer(models['STACK_0'], 'optmzr_dis_seg')
    optmzr_gen_seg = GeneratorOptimizer(models['STACK_0'], 'optmzr_gen_seg')

    optmzr_dis_img = DiscriminatorOptimizer(models['STACK_1'], 'optmzr_dis_img')
    optmzr_gen_img = GeneratorOptimizer(models['STACK_1'], 'optmzr_gen_img')

    # Pack optimizers
    optmzrs = {
        'STACK_0': {
            'optmzr_dis': optmzr_dis_seg,
            'optmzr_gen': optmzr_gen_seg, },
        'STACK_1': {
            'optmzr_dis': optmzr_dis_img,
            'optmzr_gen': optmzr_gen_img,
        }
    }

    return models, optmzrs


if __name__ == '__main__':
    [img, seg, embedding, wrong_embedding, caption] \
        = [tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3]),
           tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 90]),
           tf.placeholder(tf.float32, [BATCH_SIZE, 1024]),
           tf.placeholder(tf.float32, [BATCH_SIZE, 1024]),
           tf.placeholder(tf.string, [BATCH_SIZE, 1])]
    build_whole_graph([img, seg, embedding, wrong_embedding, caption])
