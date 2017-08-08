import tensorflow as tf
from other.config import GENERATOR_NAME, DISCRIMINATOR_IMG_NAME, DISCRIMINATOR_SEG_NAME, \
    OPTIMIZER_DIS_IMG_NAME, OPTIMIZER_DIS_SEG_NAME, OPTIMIZER_GEN_NAME


class Trainer:
    def __init__(self, sess, models, optmzrs, file_writer: tf.summary.FileWriter):
        self.sess = sess
        self.models = models
        self.optmzrs = optmzrs
        self.file_writer = file_writer
        self.merge_dict = self._merge_summaries()

    def train_gen(self):
        opt = self.optmzrs[OPTIMIZER_GEN_NAME]
        self.sess.run(opt.optmzr)


    def train_dis(self, dis_to_train):
        '''
        :param dis_to_train: DISCRIMINATOR_IMG_NAME, etc...
        '''
        opt_name = {DISCRIMINATOR_IMG_NAME: OPTIMIZER_DIS_IMG_NAME,
                    DISCRIMINATOR_SEG_NAME: OPTIMIZER_DIS_SEG_NAME}[dis_to_train]
        opt = self.optmzrs[opt_name]
        self.sess.run(opt.optmzr)

    def display(self, global_step):
        fetch = [self.merge_dict[GENERATOR_NAME],
                 self.merge_dict[DISCRIMINATOR_IMG_NAME],
                 self.merge_dict[DISCRIMINATOR_SEG_NAME],

                 self.optmzrs[OPTIMIZER_GEN_NAME].loss,
                 self.optmzrs[OPTIMIZER_DIS_IMG_NAME].loss,
                 self.optmzrs[OPTIMIZER_DIS_SEG_NAME].loss]
        res = self.sess.run(fetch)
        print('%d\tgen_loss: %.4f, dis_img_loss: %.4f, dis_seg_loss: %.4f' %(global_step, res[3], res[4], res[5]))
        self.file_writer.add_summary(res[0]+res[1]+res[2], global_step)

    def _merge_summaries(self):
        gen_merge = tf.summary.merge(
            self.models[GENERATOR_NAME].summaries +
            self.optmzrs[OPTIMIZER_GEN_NAME].summaries)
        dis_img_merge = tf.summary.merge(
            self.models[DISCRIMINATOR_IMG_NAME+'_real'].summaries +
            self.models[DISCRIMINATOR_IMG_NAME+'_fake'].summaries +
            self.optmzrs[OPTIMIZER_DIS_IMG_NAME].summaries
        )
        dis_seg_merge = tf.summary.merge(
            self.models[DISCRIMINATOR_SEG_NAME+'_real'].summaries +
            self.models[DISCRIMINATOR_SEG_NAME+'_fake'].summaries +
            self.optmzrs[OPTIMIZER_DIS_SEG_NAME].summaries
        )
        return {GENERATOR_NAME: gen_merge,
                DISCRIMINATOR_IMG_NAME: dis_img_merge,
                DISCRIMINATOR_SEG_NAME: dis_seg_merge}
