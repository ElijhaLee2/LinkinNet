import tensorflow as tf
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'


x = tf.placeholder(tf.float32, [5,1], 'x')
v = tf.Variable(3.)
tiled = tf.tile(v*x, [1,4])
y = tf.reduce_mean(tiled)

grad = tf.gradients(y,x)

z = grad+y
grad_2 = tf.gradients(z,x)

gg = tf.train.AdamOptimizer().compute_gradients(z)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    grad_res = sess.run(gg,{x:np.array([[1],[2],[3],[4],[5]])})
    print(grad_res)