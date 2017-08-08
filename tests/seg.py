import tensorflow as tf
import os
import PIL.Image as Image

SEG_PATH = "/data/bo718.wang/zhaowei/data/516data/mscoco/train2014/mask/"
os.environ['CUDA_VISIBLE_DEVICES']='6'


img_list = os.listdir(SEG_PATH)
img_queue = tf.train.string_input_producer([SEG_PATH+img for img in img_list])
img = tf.image.resize_images(tf.image.decode_jpeg(tf.WholeFileReader().read(img_queue)[1],channels=1),[64,64],method=1)
# img = tf.image.decode_jpeg(tf.WholeFileReader().read(img_queue)[1],channels=1)
one = tf.one_hot(img,56)
# max_img = tf.reduce_max(img)



sess = tf.InteractiveSession()
try:
    coord = tf.train.Coordinator()
    t = tf.train.start_queue_runners(sess,coord)
    # print(sess.run(max_img))
    img_res = sess.run(img)
    one_res = sess.run(one)
    sess.close()
finally:
    coord.request_stop()
    coord.join(t)