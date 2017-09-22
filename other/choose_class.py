# import tensorflow as tf
from pycocotools.coco import COCO
from data_input.read_data import read_data
import pickle as pkl


def main():
    CAT_NMS = ['dog']

    INSTANCES_PATH = "/data/bo718.wang/zhaowei/data/516data/mscoco/annotations/instances_train2014.json"
    coco = COCO(INSTANCES_PATH)
    name_list, embedding_list, caption_list = read_data(coco, catNms=CAT_NMS)
    # res = []
    # for tpl in zip(name_list,embedding_list,caption_list):
    #     res.append(tpl)

    f = open('/data/rui.wu/Elijha/workspace/Img_emb/dog.pkl', 'wb')
    pkl.dump([name_list, embedding_list, caption_list], f)
    f.close()


if __name__ == '__main__':
    main()
    # # a = [1.0, 'a']
    # # f = open('/data/rui.wu/Elijha/workspace/Img_emb/dset', 'wb')
    # # pkl.dump(a, f)
    # # f.close()
    #
    # f = open('/data/rui.wu/Elijha/workspace/Img_emb/dset', 'rb')
    # a = pkl.load(f)
    #
    # print()
