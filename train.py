import tensorflow as tf
from model.Trainer import Trainer
from data_input.input_pipeline import get_input_tensors
from other.config import SAVE_STEP, TOTAL_EPOCH, CUDA_VISIBLE_DEVICES, RESTORE_PATH, \
    N_DIS, SAVE_STEP_EPOCH, DISPLAY_STEP
from other.function import backup_model_file, restore_model_file
import os
import time
from model.Whole import build_whole_graph

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

# Get input tensors
batch, length = get_input_tensors()

# Build network, Create optimizer
# if RESTORE_PATH is not None:
#     # TODO shutil有延迟，因此下面建的图未必符合restore出来的设置
# TODO restore file有问题
#     restore_model_file(RESTORE_PATH)
#     time.sleep(2)  # Wait previous restore file command valid

models, optmzrs = build_whole_graph(batch)

# Create session
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)

# Prepare directories
work_dir = 'work_dir/train_64_GD_' + time.strftime("%m-%d_%H-%M-%S")
log_dir = os.path.join(work_dir, 'log')
save_path = os.path.join(work_dir, 'save')
backup_dir = os.path.join(work_dir, 'backup')
[os.makedirs(path) for path in [work_dir, log_dir, save_path, backup_dir]]

# Backup
backup_model_file(backup_dir)

# File writer % Saver
file_writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph, flush_secs=30)

saver_gan = tf.train.Saver()
saver_gan_epoch = tf.train.Saver()

# Initialization or restore
if RESTORE_PATH is None:
    tf.global_variables_initializer().run()
    # Count
    global_step = 1
    epoch_done = 0
    epoch_total = TOTAL_EPOCH
else:
    tf.global_variables_initializer().run()
    cp = tf.train.latest_checkpoint(os.path.join(RESTORE_PATH, 'save'))
    print('Restore from: ' + cp)
    saver_gan.restore(sess, cp)
    # Count
    global_step = int(cp.split('-')[-1])
    epoch_done = global_step // length
    epoch_total = TOTAL_EPOCH

# Start runners
coord = tf.train.Coordinator()
t = tf.train.start_queue_runners(sess, coord)

# Trainer
train = Trainer(sess, models, optmzrs, file_writer)

# Run
print('Run------Time: %s--------------' % time.strftime('%Y-%m-%d_%H-%M-%S'))
try:
    while epoch_done < epoch_total:
        save_flag = (global_step % SAVE_STEP == 0)
        display_flag = (global_step % DISPLAY_STEP == 0)
        n_dis = N_DIS[0 if global_step < 30 else 1]
        # train dis
        for i in range(n_dis):
            train.train_dis('dis_seg')
            # train.train_dis(DISCRIMINATOR_SEG_NAME)

        # train gen
        train.train_gen()
        if display_flag:
            train.display(global_step)

        if save_flag:
            saver_gan.save(sess, os.path.join(save_path, 'save'), global_step)

        if global_step % length == 0:
            epoch_done += 1
            if epoch_done % SAVE_STEP_EPOCH == 0:
                saver_gan_epoch.save(sess, os.path.join(save_path, 'save_epoch'), epoch_done)

        if epoch_done >= epoch_total:
            break

        global_step += 1

# except Exception as e:
#     print(e)

finally:
    print('Train finished. Total step: %d; total epoch: %d' % (global_step, epoch_done))
    coord.request_stop()
    coord.join(t)
    print('All queue runners are killed')
    sess.close()
    print('Session is closed')
