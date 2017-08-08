import tensorflow as tf
from other.function import optimize_loss
from other.config import DISCRIMINATOR_IMG_NAME, DISCRIMINATOR_SEG_NAME, GENERATOR_NAME, \
    MATCHNET_IMG_NAME, MATCHNET_SEG_NAME, MATCHNET_ADDED, SEG_ADDED
from other.hyperparameter import OPTIMIZER_DIS_HP


class DiscriminatorOptimizer:
    def __init__(self, models, net_to_train: str, name):
        '''

        :param models:
        :param net_to_train: DISCRIMINATOR_IMG, etc...
        :param name:
        '''
        with tf.name_scope(name) as scope:
            self.models = models
            self.net_to_train = net_to_train
            self.variables_to_train = self._get_variables()
            self.loss = self._get_loss()
            self.optmzr = self._get_optimizer(name)

            # self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope))
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)

    def _get_loss(self):
        lambda_ = OPTIMIZER_DIS_HP['lambda']
        dis_fake = self.models[self.net_to_train + '_fake']
        dis_real = self.models[self.net_to_train + '_real']

        # w distance
        neg_w_dis = dis_fake.score - dis_real.score
        tf.summary.scalar('neg_w_dis', -tf.reduce_mean(neg_w_dis))

        # gradient penalty
        dis_penalty = self.models[self.net_to_train + '_penalty']
        gradients = tf.gradients(dis_penalty.score, [dis_penalty.image_input])[0]
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        # tf.norm can only used for vector or matrix. In this case, each input image is a 3-D tensor, so not capable.
        # norm2 = tf.norm(gradients,ord=2, axis=[1,2,3])
        tf.summary.scalar('norm2', tf.reduce_mean(norm2))
        tf.summary.histogram('gradients_score_input', gradients)
        gradient_penalty = tf.reduce_mean(tf.square((norm2 - 1.)))

        # loss
        loss = tf.reduce_mean(neg_w_dis + lambda_ * gradient_penalty)
        return loss

    def _get_optimizer(self, name):
        learning_rate = OPTIMIZER_DIS_HP['learning_rate']
        optimizer = OPTIMIZER_DIS_HP['optimizer']
        step = tf.Variable(0, trainable=False, name=name + '/global_step')
        optmzr = optimize_loss(self.loss, step, learning_rate, optimizer,
                               variables=self.variables_to_train)
        return optmzr

    def _get_variables(self):
        model = self.models[self.net_to_train + '_real']
        return model.trainable_variables


class GeneratorOptimizer:
    def __init__(self, models, net_to_train: str, name):
        with tf.name_scope(name) as scope:
            self.models = models
            self.net_to_train = net_to_train
            self.loss = self._get_loss()
            self.variables_to_train = self._get_variables()
            self.optmzr = self._get_optimizer(name)

            # self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope))
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)

    def _get_loss(self):
        dis_fake = self.models[DISCRIMINATOR_IMG_NAME + '_fake']
        loss_dis = -tf.reduce_mean(dis_fake.score)
        tf.summary.scalar('loss_dis', loss_dis)
        loss = loss_dis

        return loss

    def _get_optimizer(self, name):
        learning_rate = OPTIMIZER_DIS_HP['learning_rate']
        optimizer = OPTIMIZER_DIS_HP['optimizer']
        step = tf.Variable(0, trainable=False, name=name + '/global_step')
        optmzr = optimize_loss(self.loss, step, learning_rate, optimizer,
                               variables=self.variables_to_train)
        return optmzr

    def _get_variables(self):
        model = self.models[GENERATOR_NAME]
        return model.trainable_variables
