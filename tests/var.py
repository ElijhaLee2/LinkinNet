import tensorflow as tf

with tf.name_scope('ns'):
    a = tf.Variable(1)
    b = tf.get_variable('b',[3,4])

with tf.variable_scope('var'):
    print(tf.get_variable_scope().name)
    c = tf.Variable(1,name='c')
    # c = tf.get_variable('c', [3, 4])

with tf.variable_scope('var'):
    print(tf.get_variable_scope().name)
    d = tf.Variable(1,name='c')
    # d = tf.get_variable('c', [3, 4])

# with tf.name_scope('ns'):

for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(v)