import tensorflow as tf


with tf.name_scope('ns'):
    v1 = tf.Variable(0,name='v1')
    print(v1.name)

    gv1 = tf.get_variable('gv1',shape=())
    print(gv1.name)

    c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ns')
    print(c)