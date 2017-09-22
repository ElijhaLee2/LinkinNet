from other.nn_func import conv


def residual_block(input_not_act, channel, norm_fn, act_fn, start_index):
    input_act = act_fn(input_not_act)
    c = input_act.get_shape().as_list()[-1]
    assert c == channel, 'Block(%d): Channel doesn\'t match! %d-%d' % (start_index, channel, c)
    # channel
    net = conv(input_act, channel // 4, 1, 1, norm_fn, act_fn, start_index)
    # channel//4
    net = conv(net, channel // 4, 3, 1, norm_fn, act_fn, start_index + 1)
    # channel//4
    net = conv(net, channel, 1, 1, norm_fn, None, start_index + 2)
    # channel
    return input_not_act + net  # no act_fn
