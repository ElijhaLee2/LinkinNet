import tensorflow.contrib.layers as ly
import tensorflow as tf
from other.function import rms, summarize
from other.config import TB_GROUP, SUM_COLLEC

# from other.hyperparameter import WEIGHT_INITIALIZER

WEIGHT_INITIALIZER = tf.truncated_normal_initializer(stddev=0.01)


def conv(inputs, channel, kernel_size, stride, normalizer_fn, activation_fn, ly_index, padding='SAME'):
    return ly.conv2d(inputs, channel, kernel_size, stride,
                     activation_fn=activation_fn, normalizer_fn=normalizer_fn, scope=str(ly_index) + '_conv',
                     weights_initializer=WEIGHT_INITIALIZER,padding=padding)


def deconv(inputs, channel, kernel_size, stride, normalizer_fn, activation_fn, ly_index):
    return ly.conv2d_transpose(inputs, channel, kernel_size, stride,
                               activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                               weights_initializer=WEIGHT_INITIALIZER, scope=str(ly_index) + '_deconv')


def deconv_with_concat(inputs, to_concat, channel, kernel_size, stride, normalizer_fn, activation_fn, ly_index):
    concat = tf.concat([inputs, to_concat], axis=3)
    return ly.conv2d_transpose(concat, channel, kernel_size, stride,
                               activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                               weights_initializer=WEIGHT_INITIALIZER, scope=str(ly_index) + '_deconv_concat')


def dense(inputs, out_num, normalizer_fn, activation_fn, ly_index):
    return ly.fully_connected(inputs, out_num, activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                              weights_initializer=WEIGHT_INITIALIZER, scope=str(ly_index) + '_fc')


def upsample_conv(inputs, channel, kernel_size, stride, normalizer_fn, activation_fn, ly_index):
    shape = inputs.get_shape().as_list()
    upsampled = tf.image.resize_bilinear(inputs, [2 * shape[1], 2 * shape[2]], name=str(ly_index) + '_resize')
    res = conv(upsampled, channel, kernel_size, stride, normalizer_fn, activation_fn, str(ly_index) + '_up_conv')
    return res


def upsample_conv_with_concat(inputs, to_concat, channel, kernel_size, stride, normalizer_fn, activation_fn, ly_index):
    shape = inputs.get_shape().as_list()
    upsampled = tf.image.resize_bilinear(inputs, [2 * shape[1], 2 * shape[2]], name=str(ly_index) + '_resize')
    concat = tf.concat([upsampled, to_concat], axis=3)
    res = conv(concat, channel, kernel_size, stride, normalizer_fn, activation_fn, str(ly_index) + '_up_conv_concat')
    return res


def leaky_relu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x, name=name + '_maximum')


# -------------------------------------------------------------------------------------------------


def optimize_loss(loss, global_step, learning_rate, optimizer, variables, g_or_d):
    """
    Summary:
        0.global:
            gradient rms: scalar
        1. variables: histogram
        2. gradients:
            2.1 gradient rms: scalar
            2.2 gradient: histogram
    """
    assert g_or_d in ['G', 'D']
    optmzr = optimizer(learning_rate)

    # Compute gradients.
    gradients = optmzr.compute_gradients(loss, variables)

    # Add global gradient rms(scalar)
    if g_or_d == 'G':
        summarize(TB_GROUP.G_grads, 'global_rms', rms(list(zip(*gradients))[0]), 'scl')
    if g_or_d == 'D':
        summarize(TB_GROUP.D_grads, 'global_rms', rms(list(zip(*gradients))[0]), 'scl')

    # Add scalar summary for loss.
    summarize(TB_GROUP.losses, g_or_d, loss, 'scl')

    # tf.summary.scalar("gradient_rms/global_rms", rms(list(zip(*gradients))[0]))


    # tf.summary.scalar("loss", loss)

    #  Add histograms for variables, gradients and gradient norms.
    for gradient, variable in gradients:
        if gradient is None:
            continue
        grad_values = gradient
        var_name = variable.name.replace(":", "_")
        if g_or_d == 'G':
            # Add summary for variables(histogram)
            # tf.summary.histogram("parameters/%s" % var_name, variable)
            summarize(TB_GROUP.G_params, var_name, variable, 'his')
            # Add summary for gradients(scalar, histogram)
            # tf.summary.scalar("gradient_rms/%s" % var_name, rms([grad_values]))
            summarize(TB_GROUP.G_grads, var_name, rms([grad_values]), 'scl')
            # tf.summary.histogram("gradients/%s" % var_name, grad_values)
            summarize(TB_GROUP.G_grads, var_name, grad_values, 'his')
        elif g_or_d == 'D':
            # Add summary for variables(histogram)
            # tf.summary.histogram("parameters/%s" % var_name, variable)
            summarize(TB_GROUP.D_params, var_name, variable, 'his')
            # Add summary for gradients(scalar, histogram)
            # tf.summary.scalar("gradient_rms/%s" % var_name, rms([grad_values]))
            summarize(TB_GROUP.D_grads, var_name, rms([grad_values]), 'scl')
            # tf.summary.histogram("gradients/%s" % var_name, grad_values)
            summarize(TB_GROUP.D_grads, var_name, grad_values, 'his')

    grad_updates = optmzr.apply_gradients(
        gradients,
        global_step=global_step,
        name="train")
    return grad_updates
