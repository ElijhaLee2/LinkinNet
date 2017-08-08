import tensorflow as tf
from model.Trainer import Trainer
from model.Discriminator import Discriminator
from model.Generator import Generator
from data_input.input_pipeline import get_input_tensors
from other.config import *

from other.function import backup_model_file, restore_model_file
import os
import time
from model.Whole import build_whole_graph

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

# Get input tensors
[img, seg, embedding], length = get_input_tensors()
g = Generator(embedding,'g')
interpolates_img = img + \
                   (tf.random_uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.) * (g.net1 - img))
d = Discriminator(interpolates_img,embedding,'d','r')

opt = tf.train.AdamOptimizer()
grad = opt.compute_gradients(d.score, d.trainable_variables)
print(grad)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

coord = tf.train.Coordinator()
t = tf.train.start_queue_runners(sess,coord)

print(sess.run(grad))

coord.request_stop()
coord.join(t)
sess.close()