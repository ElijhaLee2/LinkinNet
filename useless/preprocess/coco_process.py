# %matplotlib inline
from pycocotools.coco_data_record import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

json_path = "/data/bo718.wang/zhaowei/data/516data/mscoco/annotations/captions_train2014.json"
seg_path = "/data/bo718.wang/zhaowei/data/516data/mscoco/anns/train2014"
img_path = "/data/bo718.wang/zhaowei/data/516data/mscoco/train2014"
hdf5_path = "/data/rui.wu/CZHH/Dataset_COCO/COCO_VSE_torch/COCO_vse_torch_train.hdf5"
annotations_path = '/data/rui.wu/Elijha/annotations/'

dataDir="/data/bo718.wang/zhaowei/data/516data/mscoco/"
dataType='train2014'
annFile='%s/instances_%s.json'%(annotations_path,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n\n', ' '.join(nms))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n', ' '.join(nms))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
imgIds = coco.getImgIds(catIds=catIds );
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# load and display image
I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
# I = io.imread('http://mscoco.org/images/%d'%(img['id']))
# plt.figure(); plt.axis('off')
# plt.imshow(I)
# plt.show()

# load and display instance annotations
# plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)