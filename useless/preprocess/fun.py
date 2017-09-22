print('input')
from data_input import input_pipeline as input_pipeline

print('-------')
import os

print('tf')
import tensorflow as tf
print('-------')


print('env')
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
print('-------')

json_path = "/data/bo718.wang/zhaowei/data/516data/mscoco/annotations/captions_train2014.json"
seg_path = "/data/bo718.wang/zhaowei/data/516data/mscoco/anns/train2014"
img_path = "/data/bo718.wang/zhaowei/data/516data/mscoco/train2014"
hdf5_path = "/data/rui.wu/CZHH/Dataset_COCO/COCO_VSE_torch/COCO_vse_torch_train.hdf5"

# f = open(json_path)
# j = json.load(f)
# ann = j['annotations']

print('input_pipeline')
[img, seg, sliced_embedding_tensor, img_k, seg_k],length = input_pipeline.get_input_tensors(hdf5_path, img_path, seg_path, 128, 1)
print('-------')

print('sess')
sess = tf.InteractiveSession()
print('-------')

print('coord')
coord = tf.train.Coordinator()
t = tf.train.start_queue_runners(sess,coord)
print('-------')

print('run')
try:
    for i in range(100):
        [img_k_r,seg_k_r] = sess.run([img_k,seg_k])
        print(img_k_r[0].decode()+'\t'+seg_k_r[0].decode())
except Exception as e:
    print(e)
    print('No final?')
finally:
    sess.close()
    coord.request_stop()
    # coord.join(t)
    print('All threads killed!')
print('-------')


print()
