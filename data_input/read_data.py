from other.data_path import INSTANCES_PATH,CAP_PATH,HDF5_PATH
from other.config import CAT_NMS
from pycocotools.coco import COCO
from collections import defaultdict
import numpy as np
import json
import h5py

def read_captions(json_path):
    image_captions = defaultdict(list)
    with open(json_path) as f:
        ic_data = json.load(f)
        anns = ic_data['annotations']
        for ann in anns:
            if len(image_captions[ann['image_id']]) >= 5:  # Make sure there is only 5 captions read
                continue
            image_captions[ann['image_id']].append(ann['caption'])
    return image_captions


def read_hdf5(hdf5_path):
    return h5py.File(hdf5_path)


def imgNmToEmb(h5pyView):
    return np.array(h5pyView)


def read_cat_data(coco, catNms=[], supNms=[]):
    """
    :param cat: cat to read. if [], all cat will be read.
    :return: a list, each element of which is [img_name, seg_name, emb, cap]
    """
    assert type(catNms) == list and type(supNms) == list

    # get all cats which are the intersection of catNms and supNms, while is union in catNms and supNms
    cat_ids = coco.getCatIds(catNms=catNms, supNms=supNms)

    # img_ids
    img_ids = set()
    for cat_id in cat_ids:
        # getImgIds: intersection between each element if catIds
        img_ids |= set(coco.getImgIds(catIds=cat_id))

    # cap
    image_captions = read_captions(CAP_PATH)  # {123456:['cat is ...', ...]}
    # emb
    emb_hdf5 = h5py.File(HDF5_PATH)

    img_names = list()
    # seg_names = list()
    embs = list()
    caps = list()

    for img_id in img_ids:
        img_name = 'COCO_train2014_%.12d.jpg' % img_id
        # seg_name = img_name.replace('.jpg','.png')
        emb = imgNmToEmb(emb_hdf5[img_name])
        cap = image_captions[img_id]

        img_names.append(img_name)
        # seg_names.append(seg_name)
        embs.append(emb)
        caps.append(cap)

    return img_names, np.array(embs), caps #list of ndarray cannot be convert to tf.Tensor directly, but list of list of strings can.


if __name__ == '__main__':
    coco = COCO(INSTANCES_PATH)
    res = read_cat_data(coco, catNms=CAT_NMS)
    print(len(res))

