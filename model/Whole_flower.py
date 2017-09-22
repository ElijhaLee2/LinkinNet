import tensorflow as tf
from model.Discriminator import Discriminator
from model.Optimizer_flower import DiscriminatorOptimizer
from other.config import BATCH_SIZE, IS_STACK_1


def build_whole_graph(batch):
    [img, embedding, wrong_embedding] = batch

    models =dict()

    # Stack 1 ---------------------------------------------------
    if IS_STACK_1:
        #   dis_img -----------
        discriminator_img_real = Discriminator(img, embedding, 'dis_img', 'real')
        # discriminator_img_fake = Discriminator(generator_img.generated_pic, embedding, 'dis_img', 'fake', True)
        discriminator_img_wrong = Discriminator(img, wrong_embedding, 'dis_img', 'wrong', True)
        interpolates_emb = embedding + \
                           (tf.random_uniform(shape=[BATCH_SIZE, 1], minval=0., maxval=1.) * (
                               wrong_embedding - embedding))
        discriminator_img_penalty = Discriminator(img, interpolates_emb, 'dis_img', 'penalty', True)

        models['STACK_1'] = {
            'dis_real': discriminator_img_real,
            # 'dis_fake': discriminator_img_fake,
            'dis_wrong': discriminator_img_wrong,
            'dis_penalty': discriminator_img_penalty,}


    # Optimizer
    optmzrs = dict()

    if IS_STACK_1:
        optmzr_dis_img = DiscriminatorOptimizer(models['STACK_1'], 'optmzr_dis_img')
        # optmzr_gen_img = GeneratorOptimizer(models['STACK_1'], 'optmzr_gen_img')
        optmzrs['STACK_1'] = {
            'optmzr_dis': optmzr_dis_img,}
            # 'optmzr_gen': optmzr_gen_img, }


    return models, optmzrs


if __name__ == '__main__':
    [img, seg, embedding, wrong_embedding, caption] \
        = [tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3]),
           tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 91]),
           tf.placeholder(tf.float32, [BATCH_SIZE, 1024]),
           tf.placeholder(tf.float32, [BATCH_SIZE, 1024]),
           tf.placeholder(tf.string, [BATCH_SIZE, 1])]
    build_whole_graph([img, seg, embedding, wrong_embedding, caption])
