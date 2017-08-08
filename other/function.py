import tensorflow as tf
import os
import shutil


def leaky_relu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


# def lrelu(x, leak=0.2, name="lrelu"):
#         f1 = 0.5 * (1 + leak)
#         f2 = 0.5 * (1 - leak)
#         return f1 * x + f2 * abs(x)

# def make_multi_dirs(path_list):
#     for path in path_list:
#         os.makedirs(path)

def rms(tensor_list):
    sum = tf.Variable(0., trainable=False)
    len = tf.Variable(0., trainable=False)
    for t in tensor_list:
        # When tensor_list is a list of gradients, t is None means the corresponding variable is independent with the output
        if t is None:
            continue
        sum += tf.reduce_sum(tf.square(t))
        len += tf.reduce_prod(tf.convert_to_tensor(t.get_shape().as_list(), tf.float32))
    return tf.sqrt(sum / len)


def optimize_loss(loss, global_step, learning_rate, optimizer, variables):
    '''
    Summary:
        0.global:
            gradient rms: scalar
        1. variables: histogram
        2. gradients:
            2.1 gradient rms: scalar
            2.2 gradient: histogram
    '''
    optmzr = optimizer(learning_rate)

    # Compute gradients.
    gradients = optmzr.compute_gradients(loss, variables)

    # Add global gradient rms(scalar)
    tf.summary.scalar("gradient_rms/global_rms", rms(list(zip(*gradients))[0]))

    # Add scalar summary for loss.
    tf.summary.scalar("loss", loss)  # Add histograms for variables, gradients and gradient norms.

    for gradient, variable in gradients:
        if gradient is None:
            continue
        grad_values = gradient
        var_name = variable.name.replace(":", "_")
        # Add summary for variables(histogram)
        tf.summary.histogram("parameters/%s" % var_name, variable)
        # Add summary for gradients(scalar, histogram)
        tf.summary.scalar("gradient_rms/%s" % var_name,rms([grad_values]))
        tf.summary.histogram("gradients/%s" % var_name, grad_values)


    grad_updates = optmzr.apply_gradients(
        gradients,
        global_step=global_step,
        name="train")
    return grad_updates


def backup_model_file(backup_dir):
    shutil.copytree('./model',os.path.join(backup_dir,'model'))
    os.makedirs(os.path.join(backup_dir, 'other'),exist_ok=True)
    shutil.copy('./other/hyperparameter.py', os.path.join(backup_dir,'other', 'hyperparameter.py'))
    shutil.copy('./other/config.py', os.path.join(backup_dir,'other', 'config.py'))
    shutil.copy('./train.py', os.path.join(backup_dir, 'train.py'))


def restore_model_file(restore_dir):
    shutil.copy(os.path.join(restore_dir, 'backup', 'hyperparameter.py'), './other/hyperparameter.py')
