import tensorflow as tf
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
time.sleep(2)


def main(mode):
    if mode == 'save':
        a = tf.Variable(1.0, name='aa')
        b = tf.Variable(2.0, name='bb')

        saver = tf.train.Saver(var_list=[a, b])

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        saver.save(sess, save_path='./save')
        sess.close()
    if mode == 'restore':
        a = tf.Variable(3.0, name='aa')
        # b = tf.Variable(4.0, name='bb')
        # c = tf.Variable(5.0, name='cc')

        saver = tf.train.Saver()
        # the default var_list of Saver is all variables in this CODE_DEFINED graph (not the saved graph in checkpoint);
        # if there is any variable in var_list that cannot be found in checkpoint when 'restore',
        # an error will be raised.

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        re_path = tf.train.latest_checkpoint('./')
        saver.restore(sess, save_path=re_path)
        print(a.eval())
        # print(b.eval())
        # print(c.eval())
        sess.close()


if __name__ == '__main__':
    # main('save')
    main('restore')
