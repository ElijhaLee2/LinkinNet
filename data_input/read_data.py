from other.data_path import INSTANCES_PATH, CAP_PATH, HDF5_PATH
from other.config import CAT_NMS
from pycocotools.coco import COCO
from collections import defaultdict
import numpy as np
import json
import h5py
import pickle

emb_mean = 0.0028702784
emb_var = 0.14780775


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
    # return (np.array(h5pyView) - emb_mean) / emb_var
    return np.array(h5pyView)


def read_data(coco, catNms=[], supNms=[]):
    """
    if both catNums and supNms are empty, read all imgs in mscoco
    :param cat: cat to read. if [], all cat will be read.
    :return: a list, each element of which is [img_name, seg_name, emb, cap]
    """
    assert type(catNms) == list and type(supNms) == list

    if len(catNms) == 0 and len(supNms) == 0:
        img_ids = coco.imgs.keys()
    else:
        cat_ids = coco.getCatIds(catNms=catNms, supNms=supNms)
        img_ids = set()
        for cat_id in cat_ids:
            # getImgIds: intersection between each element if catIds
            img_ids |= set(coco.getImgIds(catIds=cat_id))
    print('Pic num in training set: %d' % len(img_ids))

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
        emb = imgNmToEmb(emb_hdf5[img_name])  # 5,1024
        cap = image_captions[img_id]

        img_names.append(img_name)
        # seg_names.append(seg_name)
        embs.append(emb)
        caps.append(cap)

    print('catNms=%s, supNms=%s' % (str(catNms), str(supNms)))
    return img_names, np.array(
        embs), caps  # list of ndarray cannot be convert to tf.Tensor directly, but list of list of strings can.


def read_picked_data(pkl_path):
    f = open(pkl_path, 'rb')
    res = pickle.load(f)
    # cap = ['%d: %s' % (i, c) for i, c in enumerate(res[2])]
    # for i,c_list in enumerate(res[2]):
    #     for c in c_list:
    #         c = '%d: %s'%(i,c)
    print('Total data number: ', len(res[0]))
    return res[0], res[1], res[2]


if __name__ == '__main__':
    coco = COCO(INSTANCES_PATH)
    res = read_data(coco, catNms=CAT_NMS)
    print(len(res))
