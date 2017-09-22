import tensorflow as tf
from other.config import SUM_COLLEC


class Trainer:
    def __init__(self, sess, models, optmzrs, file_writer: tf.summary.FileWriter):
        self.sess = sess
        self.models = models
        self.optmzrs = optmzrs
        self.file_writer = file_writer
        self.merge = self._merge_summaries()

    def train_gen(self, stack_to_train):
        opt = self.optmzrs[stack_to_train]['optmzr_gen']
        self.sess.run(opt.optmzr)

    def train_dis(self, stack_to_train):
        opt = self.optmzrs[stack_to_train]['optmzr_dis']
        self.sess.run(opt.optmzr)

    def display(self, global_step):
        fetch = self.merge
        res = self.sess.run(fetch)
        self.file_writer.add_summary(res, global_step)
        print(global_step)

    def _merge_summaries(self):
        summaries = list()
        for k in self.models.keys():

            for m in self.models[k].values():
                summaries += m.summaries

            for o in self.optmzrs[k].values():
                summaries += o.summaries

        return tf.summary.merge(summaries)
