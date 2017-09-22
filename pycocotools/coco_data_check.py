from pycocotools.coco import COCO
CAP_PATH = "/data/bo718.wang/zhaowei/data/516data/mscoco/annotations/captions_train2014.json"
INSTANCES_PATH = "/data/bo718.wang/zhaowei/data/516data/mscoco/annotations/instances_train2014.json"

coco = COCO(INSTANCES_PATH)

# Count img number of cat='person'
person_id = coco.getCatIds(catNms=['bird'])
img_ids = coco.getImgIds(catIds=person_id)
print(len(img_ids))


person_id = coco.getCatIds(supNms=['animal'])
img_ids = coco.getImgIds(catIds=person_id)
print(len(img_ids))

person_id = coco.getCatIds(catNms=['person'],supNms=['person'])
img_ids = coco.getImgIds(catIds=person_id)
print(len(img_ids))