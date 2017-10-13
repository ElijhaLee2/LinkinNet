# import tensorflow as tf
# import numpy as np
#
#
# def norm(tensor):
#     return tensor / tf.reduce_sum(tensor)
#     # return tensor / tensor
#
#
# if __name__ == '__main__':
#     a = [1., 2., 3.]
#     a = np.array(a)
#
#     t = tf.placeholder(tf.float32)
#     res = norm(t)
#
#     with tf.Session() as sess:
#         print(sess.run(res, {t: a}))
