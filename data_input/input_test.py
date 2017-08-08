import tensorflow as tf
from data_input.input_pipeline import get_input_tensors
from other.config import SAVE_STEP, TOTAL_EPOCH, CUDA_VISIBLE_DEVICES, RESTORE_PATH, MATCHNET_IMG_NAME
import os
import time
import numpy as np
import PIL.Image as Image

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
time.sleep(2)




# Get input tensors
[img, seg, embedding, caption], length = get_input_tensors(is_debug=True)

# Summary: img
tf.summary.image('img', img, max_outputs=img.get_shape().as_list()[0])

# Summary: seg
trans_seg = tf.expand_dims(tf.arg_max(seg,3),3)
tf.summary.image('seg',tf.cast(trans_seg,tf.float32), max_outputs=trans_seg.get_shape().as_list()[0])

# Summary: img * seg
mul_res = img * tf.cast(tf.cast(tf.tile(tf.expand_dims(tf.arg_max(seg,3),3),[1,1,1,3]),tf.bool),tf.float32)

tf.summary.image('seg_img', mul_res, max_outputs=mul_res.get_shape().as_list()[0])

merge = tf.summary.merge_all()

coord = tf.train.Coordinator()

with tf.Session() as sess, open('caption','w') as f:
    fw = tf.summary.FileWriter('./log',sess.graph)
    tf.train.start_queue_runners(sess,coord)

    while 1:
        try:
            merge_res,cap = sess.run([merge, caption])
            fw.add_summary(merge_res)
            fw.flush()
            # f.write(cap)
        finally:
            print('hhh')
            coord.request_stop()


