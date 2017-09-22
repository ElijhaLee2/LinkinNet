from enum import Enum

CUDA_VISIBLE_DEVICES = '5'
print('CUDA_VISIBLE_DEVICES: ', CUDA_VISIBLE_DEVICES)

ALLOW_GROWTH = True

WORK_DIR_NAME = 'stack_plus'

BATCH_SIZE = 48
IMG_LENGTH = 64

#
SAVE_STEP = 100
SAVE_STEP_EPOCH = 50
TOTAL_EPOCH = 600

# MSCOCO class
N_CAT = 90 + 1
CAT_NMS = ['dog']
CAT_NUMs = [18]

# IS_DEBUG = True
IS_DEBUG = False

print('IS_DEBUG: ', IS_DEBUG)

if not IS_DEBUG:
    DISPLAY_STEP = 10
    N_DIS = [30, 5]
else:
    DISPLAY_STEP = 1
    N_DIS = [1, 1]

# IS_RESTORE = True
IS_RESTORE = False
print('IS_RESTORE: ', IS_RESTORE)

IS_STACK_0 = False
IS_STACK_1 = True
print('(%s, %s)' % (IS_STACK_0, IS_STACK_1))


# TB_GROUP = Enum('TB_GROUP',
#                 ('G_params', 'D_params', 'G_grads', 'D_grads', 'scores', 'outputs', 'norms', 'w_dis', 'losses',
#                  'gen_img'))


class TB_GROUP:
    G_params = 'G_params'
    D_params = 'D_params'
    G_grads = 'G_grads'
    D_grads = 'D_grads'
    scores = 'scores'
    # outputs = 'outputs'
    norms = 'norms'
    w_dis = 'w_dis'
    losses = 'losses'
    # gen_img = 'gen_img'


# SUM_COLLEC = Enum('SUM_COLLEC', ('G_sum', 'D_sum', 'Opt_G_sum', 'Opt_D_sum'))

class SUM_COLLEC:
    G_sum = 'G_sum'
    D_sum = 'D_sum'
    Opt_G_sum = 'Opt_G_sum'
    Opt_D_sum = 'Opt_D_sum'
