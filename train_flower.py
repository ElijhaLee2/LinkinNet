import tensorflow as tf
from model.Trainer import Trainer
from data_input.input_pipeline_flower import get_input_tensors
from other.config import SAVE_STEP, TOTAL_EPOCH, CUDA_VISIBLE_DEVICES, IS_RESTORE, \
    N_DIS, SAVE_STEP_EPOCH, DISPLAY_STEP, IS_STACK_0, IS_STACK_1, WORK_DIR_NAME
from other.function import backup_model_file, get_trained_num
import os
import sys
import time
from model.Whole_flower import build_whole_graph

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

# Get input tensors
batch, length = get_input_tensors()

# Build network, Create optimizer
models, optmzrs = build_whole_graph(batch)

# Create session
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)

# Prepare directories
if IS_RESTORE:
    log_dir = os.path.join('..', 'log')
    save_path = os.path.join('..', 'save')
else:
    name_run = 'flower_rw'
    while not ((input('Name of this run is %s, confirm?' % name_run)) in ['y', 'Y', 'yes']):
        name_run = input('Re-type in the name of this run:')
    work_dir = os.path.join('work_dir', '%s_%s' % (name_run, time.strftime("%m-%d_%H-%M-%S")))
    log_dir = os.path.join(work_dir, 'log')
    save_path = os.path.join(work_dir, 'save')
    backup_dir = os.path.join(work_dir, 'backup')
    [os.makedirs(path, exist_ok=True) for path in [work_dir, log_dir, save_path, backup_dir]]
    # Backup
    backup_model_file(backup_dir)

# File writer % Saver
file_writer = tf.summary.FileWriter(logdir=log_dir,
                                    # graph=sess.graph if not IS_RESTORE else None,
                                    flush_secs=30)

saver_gan = tf.train.Saver()
saver_gan_epoch = tf.train.Saver()

# Initialization or restore
if not IS_RESTORE:
    tf.global_variables_initializer().run()
    # Count
    global_step = 1
    epoch_done = 0
    epoch_total = TOTAL_EPOCH
else:
    tf.global_variables_initializer().run()
    cp = tf.train.latest_checkpoint(os.path.join(sys.path[0], '../save'))
    y_or_n = input('Restore from: %s, true? (\'y\' for yes, others for no.)' % cp)
    assert (y_or_n == 'y' or y_or_n == 'Y'), 'Not \'y\' or \'Y\', exit.'

    saver_gan.restore(sess, cp)
    # Count
    global_step = int(cp.split('-')[-1])
    epoch_done = get_trained_num(global_step) // length
    epoch_total = TOTAL_EPOCH

# Start runners
coord = tf.train.Coordinator()
t = tf.train.start_queue_runners(sess, coord)
queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
# print('Queue runners: ')
# [print(qr.name, end='\n\n') for qr in queue_runners]

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
        if IS_STACK_0:
            train.train_dis('STACK_0')
        if IS_STACK_1:
            train.train_dis('STACK_1')


        # display
        if display_flag:
            train.display(global_step)

        if save_flag:
            saver_gan.save(sess, os.path.join(save_path, 'save'), global_step)

        if get_trained_num(global_step) % length == 0:
            epoch_done += 1
            if epoch_done % SAVE_STEP_EPOCH == 0:
                saver_gan_epoch.save(sess, os.path.join(save_path, 'save_epoch'), epoch_done)

        if epoch_done >= epoch_total:
            break

        global_step += 1


finally:
    print('Train finished. Total step: %d; total epoch: %d' % (global_step, epoch_done))
    coord.request_stop()
    coord.join(t)
    print('All queue runners are killed')
    sess.close()
    print('Session is closed')
