from pycocotools.coco import COCO
CAP_PATH = "/data/bo718.wang/zhaowei/data/516data/mscoco/annotations/captions_train2014.json"
INSTANCES_PATH = "/data/bo718.wang/zhaowei/data/516data/mscoco/annotations/instances_train2014.json"

coco = COCO(INSTANCES_PATH)
print()

print('cat\tsupercat\tcount')
for c in coco.cats:
    cc = coco.loadCats(c)
    print('%s\t%s\t%d'%(cc[0]['name'], cc[0]['supercategory'],len(coco.getImgIds(catIds=c))))