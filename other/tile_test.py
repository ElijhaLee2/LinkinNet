import tensorflow as tf
import os,time
import numpy as np



def tile(emb, pic_size: list):
    mul = pic_size[0] * pic_size[1]
    [batch_size, channel] = emb.get_shape().as_list()

    res = [emb] * mul
    res = tf.stack(res, axis=1)
    res = tf.reshape(res, [batch_size, pic_size[0], pic_size[1], channel])

    return res


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES']='1'
    time.sleep(2)

    x = tf.placeholder(tf.int32,shape=[1,2])
    z = tile(x,[2,2])
    z1 = tf.tile(tf.expand_dims(tf.expand_dims(x,1),1),[1,2,2,1])
    print(z.get_shape().as_list())

    sess = tf.Session()
    z_r, z1_r = sess.run([z,z1],{x:np.array([[2,3]])})
    print(z_r)
    print(z1_r)
    print(z_r==z1_r)