from pycocotools.coco import *

JSON_PATH = "/data/bo718.wang/zhaowei/data/516data/mscoco/annotations/instances_train2014.json"

coco = COCO(JSON_PATH)
catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
imgIds = coco.getImgIds(catIds=catIds );
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)