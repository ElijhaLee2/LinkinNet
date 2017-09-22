import tensorflow as tf




def random_crop(value, size, seed=None, name=None):
    with tf.name_scope(name, "random_crop", [value, size]) as name:
        channel = value.get_shape().as_list()[-1]
        value = tf.convert_to_tensor(value, name="value")
        size = tf.convert_to_tensor(size, dtype=tf.int32, name="size")
        shape = tf.shape(value)[0:2]
        limit = shape - size + 1
        offset = tf.random_uniform(
            tf.shape(shape),
            dtype=size.dtype,
            maxval=size.dtype.max,
            seed=seed) % limit

        ret = [tf.slice(value[:,:,c], offset, size) for c in range(channel)]


        return tf.stack(ret,axis=2)


if __name__ == '__main__':
    img = tf.placeholder(tf.uint8, [200, 200, 3])

    shape = tf.cast(tf.shape(img), tf.float32)
    min_edge = tf.minimum(shape[0], shape[1])
    w = h = tf.cast(0.8 * min_edge, tf.int32)

    # The cropped size can be determined only when each dimension in size is determined
    crop2 = random_crop(img, [h, w])  # random crop can receive tensor as size~


    print()