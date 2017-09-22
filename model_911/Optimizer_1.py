import tensorflow as tf

from model_911.hyperparameter import OPTIMIZER_DIS_HP, OPTIMIZER_GEN_HP
from other.config import BATCH_SIZE, TB_GROUP, SUM_COLLEC, N_CAT
from other.function import l2_norm_square, summarize
from other.nn_func import optimize_loss


class DiscriminatorOptimizer:
    def __init__(self, stack_to_train, name):
        with tf.name_scope(name) as scope:
            self.stack_to_train = stack_to_train
            self.variables_to_train = self._get_variables()
            self.loss = self._get_loss()
            self.optmzr = self._get_optimizer(name)
            self._summary_dis()

            # self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope))
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)

    def _get_loss(self):
        lambda_ = OPTIMIZER_DIS_HP['lambda']
        dis_fake = self.stack_to_train['dis_fake']
        dis_real = self.stack_to_train['dis_real']
        dis_wrong = self.stack_to_train['dis_wrong']
        dis_gp_rf = self.stack_to_train['dis_gp_rf']
        dis_gp_rw = self.stack_to_train['dis_gp_rw']

        # summary w distance
        summarize(TB_GROUP.w_dis, 'rf', tf.reduce_mean(dis_real.score - dis_fake.score), 'scl')
        summarize(TB_GROUP.w_dis, 'rw', tf.reduce_mean(dis_real.score - dis_wrong.score), 'scl')

        gp_rf = self.compute_gp(dis_gp_rf, 'rf_gp')
        gp_rw = self.compute_gp(dis_gp_rw, 'rw_gp')

        loss = (tf.reduce_mean(dis_fake.score) + tf.reduce_mean(dis_wrong.score)) \
               - tf.reduce_mean(dis_real.score) * 2 \
               + lambda_ * (gp_rf + gp_rw)

        return loss

    def _get_optimizer(self, name):
        learning_rate = OPTIMIZER_DIS_HP['learning_rate']
        optimizer = OPTIMIZER_DIS_HP['optimizer']
        step = tf.Variable(0, trainable=False, name=name + '/global_step')
        optmzr = optimize_loss(self.loss, step, learning_rate, optimizer,
                               self.variables_to_train, 'D')
        ret = optmzr

        return ret

    def _get_variables(self):
        model = self.stack_to_train['dis_real']
        return model.trainable_variables

    def _summary_dis(self, ):
        dis_real = self.stack_to_train['dis_real']
        dis_fake = self.stack_to_train['dis_fake']
        real_pic = self.one_hot_inverse(dis_real.image_input) if dis_real.variable_scope_name.find(
            'seg') != -1 else dis_real.image_input[:, :, :, 0:3]
        fake_pic = self.one_hot_inverse(dis_fake.image_input) if dis_fake.variable_scope_name.find(
            'seg') != -1 else dis_fake.image_input[:, :, :, 0:3]
        # TODO just for 'dog'
        concat = tf.concat(
            [tf.concat([dis_real.image_input[:, :, :, 0:3], fake_pic], axis=2),
             tf.concat([dis_real.image_input[:, :, :, 3:6], dis_real.image_input[:, :, :, 6:9]], axis=2)],
            axis=1)

        # concat = tf.concat([real_pic, fake_pic], axis=2)
        summarize('gen_img', 'r_f', concat, 'img')
        # tf.summary.image('real_fake', concat, max_outputs=BATCH_SIZE)
        # tf.summary.histogram('real_pic', real_pic)
        summarize('out_his', 'real_pic', real_pic, 'his')
        # tf.summary.histogram('fake_pic', fake_pic)
        summarize('out_his', 'fake_pic', fake_pic, 'his')

    def one_hot_inverse(self, pic):
        return \
            tf.cast(
                tf.expand_dims(
                    tf.arg_max(pic, -1),
                    axis=3),
                tf.float32)

    def compute_gp(self, dis_penalty, gp_name):
        with tf.name_scope(gp_name):
            [gradients_img, gradients_seg, gradients_emb] = \
                tf.gradients(dis_penalty.score,
                             [dis_penalty.image_input, dis_penalty.seg_input, dis_penalty.emb_input],
                             name='dis_gp_grad')

            norm_img_square = l2_norm_square(gradients_img)
            norm_seg_square = l2_norm_square(gradients_seg)
            norm_emb_square = l2_norm_square(gradients_emb)

            norm2 = tf.sqrt(norm_img_square + norm_seg_square + norm_emb_square)
            norm2_img = tf.sqrt(norm_img_square)
            norm2_seg = tf.sqrt(norm_seg_square)
            norm2_emb = tf.sqrt(norm_emb_square)

            # # tf.summary.histogram('gradients_img', gradients_img)
            # summarize('other_grad', '%s/gradients_img' % gp_name, gradients_img, 'his')
            # # tf.summary.histogram('gradients_emb', gradients_emb)
            # summarize('other_grad', '%s/gradients_emb' % gp_name, gradients_emb, 'his')

            summarize(TB_GROUP.norms, gp_name + '/' + 'norm', tf.reduce_mean(norm2), 'scl')
            summarize(TB_GROUP.norms, gp_name + '/' + 'norm_img', tf.reduce_mean(norm2_img), 'scl')
            summarize(TB_GROUP.norms, gp_name + '/' + 'norm_seg', tf.reduce_mean(norm2_seg), 'scl')
            summarize(TB_GROUP.norms, gp_name + '/' + 'norm_emb', tf.reduce_mean(norm2_emb), 'scl')

        return tf.reduce_mean(tf.square(norm2 - 1))


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
        # tf.summary.scalar('loss_dis', loss_dis)
        loss = loss_dis

        return loss

    def _get_optimizer(self, name):
        learning_rate = OPTIMIZER_GEN_HP['learning_rate']
        optimizer = OPTIMIZER_GEN_HP['optimizer']
        step = tf.Variable(0, trainable=False, name=name + '/global_step')
        optmzr = optimize_loss(self.loss, step, learning_rate, optimizer,
                               self.variables_to_train, 'G')
        return optmzr

    def _get_variables(self):
        model = self.stack_to_train['gen']
        return model.trainable_variables
