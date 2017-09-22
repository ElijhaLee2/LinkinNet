import json
import h5py
import numpy as np

HDF5_PATH = "/data/rui.wu/CZHH/Dataset_COCO/COCO_VSE_torch/COCO_vse_torch_train.hdf5"

hdf5_path = HDF5_PATH
h = h5py.File(hdf5_path)
f = open('COCO_vse.json','w')


name_emb = {}
cnt = 0
for hh in h.items():
    emb = np.array(hh[1])[0:5]
    name_emb[int(hh[0][21:27])] = emb.tolist()
    print(cnt)
    cnt += 1


json.dump(name_emb,f)
f.flush()
f.close()
