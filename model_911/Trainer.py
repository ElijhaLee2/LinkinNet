import tensorflow as tf


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

    def display(self, global_step, total_step, time_inter):
        fetch = self.merge
        res = self.sess.run(fetch)
        self.file_writer.add_summary(res, global_step)
        print('%d (%.4f%%): , %.3f' % (global_step, global_step / total_step, time_inter))

    def _merge_summaries(self):
        summaries = tf.get_collection('LinkNet_sum')
        return tf.summary.merge(summaries)
