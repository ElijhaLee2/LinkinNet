import tensorflow as tf
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
time.sleep(2)

a = tf.get_variable('a', shape=(1000,))
b = tf.get_variable('b', shape=(1000,))

with tf.name_scope('a'):
    tf.summary.histogram('a', a)
with tf.name_scope('b'):
    tf.summary.histogram('b', b)

m_a = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,'a'))
m_b = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,'b'))

# m = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,'a')+tf.get_collection(tf.GraphKeys.SUMMARIES,'b'))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
fw = tf.summary.FileWriter('.',sess.graph)
m_res1 = sess.run(m_a)
m_res2 = sess.run(m_b)
fw.add_summary(m_res1+m_res2,1)
fw.flush()
sess.close()