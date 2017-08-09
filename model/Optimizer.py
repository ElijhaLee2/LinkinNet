import tensorflow as tf
from other.function import optimize_loss
from other.hyperparameter import OPTIMIZER_DIS_HP,OPTIMIZER_GEN_HP


class DiscriminatorOptimizer:
    def __init__(self, stack_to_train, name):
        '''

        :param models:
        :param stack_to_train: a dict consist of g, dis_real, ...
        :param name:
        '''
        with tf.name_scope(name) as scope:
            self.stack_to_train = stack_to_train
            self.variables_to_train = self._get_variables()
            self.loss = self._get_loss()
            self.optmzr = self._get_optimizer(name)

            # self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope))
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)

    def _get_loss(self):
        lambda_ = OPTIMIZER_DIS_HP['lambda']
        dis_fake = self.stack_to_train['dis_fake']
        dis_real = self.stack_to_train['dis_real']
        dis_wrong = self.stack_to_train['dis_wrong']
        dis_penalty = self.stack_to_train['dis_penalty']

        # w distance
        neg_w_dis = dis_fake.score - dis_real.score
        tf.summary.scalar('neg_w_dis', -tf.reduce_mean(neg_w_dis))

        # gradient penalty
        gradients = tf.gradients(dis_penalty.score, [dis_penalty.image_input])[0]
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        # tf.norm can only used for vector or matrix. In this case, each input image is a 3-D tensor, so not capable.
        # norm2 = tf.norm(gradients,ord=2, axis=[1,2,3])
        tf.summary.scalar('norm2', tf.reduce_mean(norm2))
        tf.summary.histogram('gradients_score_input', gradients)
        gradient_penalty = tf.reduce_mean(tf.square((norm2 - 1.)))

        # loss
        loss = tf.reduce_mean((dis_fake.score + dis_wrong.score) / 2 - dis_real.score + lambda_ * gradient_penalty)
        return loss

    def _get_optimizer(self, name):
        learning_rate = OPTIMIZER_DIS_HP['learning_rate']
        optimizer = OPTIMIZER_DIS_HP['optimizer']
        step = tf.Variable(0, trainable=False, name=name + '/global_step')
        optmzr = optimize_loss(self.loss, step, learning_rate, optimizer,
                               variables=self.variables_to_train)
        return optmzr

    def _get_variables(self):
        model = self.stack_to_train['dis_real']
        return model.trainable_variables


class GeneratorOptimizer:
    def __init__(self, stack_to_train, name):
        with tf.name_scope(name) as scope:
            self.stack_to_train = stack_to_train
            self.variables_to_train = self._get_variables()
            self.loss = self._get_loss()
            self.optmzr = self._get_optimizer(name)

            # self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope))
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)

    def _get_loss(self):
        dis_fake = self.stack_to_train['dis_fake']
        loss_dis = -tf.reduce_mean(dis_fake.score)
        tf.summary.scalar('loss_dis', loss_dis)
        loss = loss_dis

        return loss

    def _get_optimizer(self, name):
        learning_rate = OPTIMIZER_GEN_HP['learning_rate']
        optimizer = OPTIMIZER_GEN_HP['optimizer']
        step = tf.Variable(0, trainable=False, name=name + '/global_step')
        optmzr = optimize_loss(self.loss, step, learning_rate, optimizer,
                               variables=self.variables_to_train)
        return optmzr

    def _get_variables(self):
        model = self.stack_to_train['gen']
        return model.trainable_variables
