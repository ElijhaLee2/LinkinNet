# If this file is not in the root dir of this project, when `import COCO` it will raise exception like 'json has no load()', so I have to put this file under root

# %matplotlib inline

import sys
import os
import time

for p in sys.path:
    print(p)

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = '/home/elijha/Documents/Data/MSCOCO'
dataType = 'train2014'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
seg_target = '/home/elijha/Documents/Data/MSCOCO/train2014_seg'

# initialize COCO api for instance annotations
coco = COCO(annFile)


# # # display COCO categories and supercategories
# # cats = coco.loadCats(coco.getCatIds())
# # nms=[cat['name'] for cat in cats]
# # print('COCO categories: \n{}\n'.format(' '.join(nms)))
# #
# # nms = set([cat['supercategory'] for cat in cats])
# # print('COCO supercategories: \n{}'.format(' '.join(nms)))
#
# # get all images containing given categories, select one at random
# # catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
# # imgIds = coco.getImgIds(catIds=catIds )
# imgIds = coco.getImgIds(imgIds=[262145])
# img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
#
# # load and display image
# # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# # use url to load image
# I = io.imread(img['coco_url'])
# plt.figure(0)
# plt.axis('off')
# plt.imshow(I)
# # plt.show()
#
# # load and display instance annotations
# # plt.imshow(I); plt.axis('off')
# annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)
# plt.show()


def _get_mask_array(anns, shape: list):
    # each ann in anns is a instance.
    mask = np.zeros(shape, np.uint8)
    for ann in anns:
        tmp = coco.annToMask(ann)
        mask_mask = mask == 0
        mask += mask_mask * tmp * ann['category_id']
    return mask


def get_mask_arr_by_name(img_name):
    img_id = int(img_name[15:-4])
    anns = coco.imgToAnns[img_id]
    img_ = coco.loadImgs(img_id)[0]
    mask_arr = _get_mask_array(anns, [img_['height'], img_['width']])
    return mask_arr


def plot_img(img_arr, img_no):
    plt.figure(img_no)
    plt.imshow(img_arr)
    plt.show()


img_list = os.listdir(os.path.join(dataDir, dataType))
cnt = 0
total = len(img_list)
for img_name in img_list:
    tic = time.time()
    # COCO_train2014_000000004587
    mask_arr = get_mask_arr_by_name(img_name)
    io.imsave(os.path.join(seg_target, img_name.replace('.jpg', '.png')), mask_arr)
    cnt += 1
    if cnt % 100 == 0:
        toc = time.time()
        dif = toc - tic
        rem = (total - cnt) * dif
        print("%d; remain: %s" % (cnt, rem))
