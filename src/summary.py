import tensorflow as tf


def view_activation(name, layer_op):
    # ":1" to just view the first samples' layers.
    with tf.name_scope('summary/activation/'):
        c_layer_op = tf.transpose(layer_op[:1], (3, 1, 2, 0))
        c_layer_op = tf.to_float(c_layer_op)

    return tf.summary.image(name, c_layer_op, 30)


def weight_summary(grad_var_op):
    summaries = list()

    for dx, x in grad_var_op:
        if dx is None:
            continue
        elif 'batch_normalization' in x.name:
            continue

        summaries += [tf.summary.histogram('weight/' + x.name, x)]

    return summaries


def gradient_summary(grad_var_op, learn_rate_op, eps=1e-7):
    summaries = list()

    with tf.name_scope('summary/gradients'):
        for dx, x in grad_var_op:
            if dx is None:
                continue
            elif 'batch_normalization' in x.name:
                continue

            rel_dx = learn_rate_op * tf.abs(tf.div(dx, x + eps))

            summaries += [tf.summary.histogram('dx/' + x.name, rel_dx)]

    return summaries
