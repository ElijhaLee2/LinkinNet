OPTIMIZER_SUMMARIES = [
    "learning_rate",
    "loss",
    # "gradients",
    # "gradient_norm",
]

CUDA_VISIBLE_DEVICES = '3'

RESTORE_PATH = None
# RESTORE_PATH = '/data/rui.wu/Elijha/workspace/Img_emb/work_dir/train08-03_10-40-44/'

# Path of MatchNet saved parameters
MATCHNET_IMG_SAVE_PATH = '/data/rui.wu/Elijha/workspace/Img_emb/params/img/'
MATCHNET_SEG_SAVE_PATH = '/data/rui.wu/Elijha/workspace/Img_emb/params/seg/'

# # Names
# DISCRIMINATOR_IMG_NAME = 'dis_img'
# DISCRIMINATOR_SEG_NAME = 'dis_seg'
# GENERATOR_NAME = 'gen'
# OPTIMIZER_GEN_NAME = 'optmzr_gen'
# OPTIMIZER_DIS_IMG_NAME = 'optmzr_dis_img'
# OPTIMIZER_DIS_SEG_NAME = 'optmzr_dis_seg'


BATCH_SIZE = 32
IMG_SIZE = 256

# IS_DEBUG = True
IS_DEBUG = False
if not IS_DEBUG:
    DISPLAY_STEP = 5
    N_DIS = [20,10]
else:
    DISPLAY_STEP = 1
    N_DIS = [3,1]

SAVE_STEP = 100
SAVE_STEP_EPOCH = 5
TOTAL_EPOCH = 20

# MSCOCO class number
N_CLASS = 90

SEG_ADDED = True
MATCHNET_ADDED = True