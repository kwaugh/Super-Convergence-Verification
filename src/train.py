import os
import argparse

import numpy as np
import tensorflow as tf

import config
import model
import ops
import run
import grade


slim = tf.contrib.slim


def get_train_op(loss_op, train_vars):
    bounds = [50000, 100000]
    values = [2e-2, 1e-2, 1e-3]

    with tf.variable_scope('training'):
        step_op = tf.Variable(0, name='step', trainable=False)

        learn_rate_op = tf.train.piecewise_constant(step_op, bounds, values)
        optimizer_op = tf.train.AdamOptimizer(learn_rate_op)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            grad_var_op = optimizer_op.compute_gradients(
                    loss_op, var_list=train_vars)

            train_op = optimizer_op.apply_gradients(
                    grad_var_op, global_step=step_op)

    return train_op, learn_rate_op, step_op, grad_var_op


def get_inputs(runner_train, runner_valid, batch_size):
    with tf.name_scope('is_training'):
        is_training_op = tf.placeholder(tf.bool)

    with tf.name_scope('samples'):
        inputs_list = tf.cond(
                is_training_op,
                lambda: runner_train.get_inputs(batch_size),
                lambda: runner_valid.get_inputs(batch_size))

    return is_training_op, inputs_list


def load_checkpoint(sess, checkpoint, scope):
    slim.assign_from_checkpoint_fn(checkpoint, slim.get_model_variables(scope))(sess)


def train_loop(runner_train, runner_valid,
               step_op, train_op, summary_op, is_training_op,
               saved_vars, save_path, log_dir, ensemble,
               pre_loop_callback=lambda x: None):
    saved_vars_dict = {key.name: key for key in saved_vars}
    saver = tf.train.Saver(saved_vars_dict)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()

        threads = list()
        threads += runner_train.create_threads(sess, coord)
        threads += runner_valid.create_threads(sess, coord)

        # Allows for loading of variables.
        pre_loop_callback(sess)

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        try:
            best_accuracy = -1.0

            for _ in range(config.num_steps):
                if coord.should_stop():
                    break

                step = sess.run(step_op)

                if step % config.checkpoint_steps == 0:
                    summary_train = sess.run(summary_op, {is_training_op: True})
                    summary_valid = sess.run(summary_op, {is_training_op: False})

                    summary_writer.add_summary(summary_train, step)
                    summary_writer.add_summary(summary_valid, step)

                if step % config.save_steps == 0:
                    ensemble.sess = sess
                    run.Predictor.sun_network = ensemble

                    print('Step: %d' % step)
                    tmp_best_accuracy = grade.evaluate(run.Predictor)

                    if tmp_best_accuracy > best_accuracy:
                        best_accuracy = tmp_best_accuracy
                        saver.save(sess, save_path)

                sess.run(train_op, {is_training_op: True})

            coord.request_stop()
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()

            saver.save(sess, save_path, global_step=step)
            print('Saved to %s at step %d.' % (save_path, step_op.eval()))
            coord.request_stop(e)
        finally:
            coord.join(threads)


def main():
    save_path = os.path.join(config.log_dir, config.model_name)

    runner_train, runner_valid = model.SunNet.get_runners(
            config.input_shape, config.num_classes)

    is_training_op, (images_op, labels_op) = get_inputs(
            runner_train, runner_valid, config.batch_size)

    # data augmentation
    images_op = tf.cond(is_training_op,
                        lambda: ops.batch_data_augment(images_op),
                        lambda: images_op)
    # no data augment
    # images_op = tf.cond(is_training_op,
                        # lambda: images_op,
                        # lambda: images_op)

    network = model.SunNetRunner(save_path, config.num_classes,
                                    images_op, labels_op, is_training_op)

    loss_op = network.loss_op
    pred_op = network.pred_op

    # Used for saving and gradient descent.
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SunNet')

    print('Weights to be trained.')
    print('\n'.join(sorted(map(lambda x: x.name, train_vars))))
    print()

    train_op, learn_rate_op, step_op, grad_var_op = get_train_op(loss_op, train_vars)

    # Includes moving averages and such.
    saved_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SunNet')
    saved_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'training')

    print('Weights to be saved.')
    print('\n'.join(sorted(map(lambda x: x.name, saved_vars))))

    summary_op = model.SunNet.get_summaries(
            images_op, labels_op, pred_op,
            config.num_classes, learn_rate_op, loss_op,
            grad_var_op, is_training_op, network)

    train_loop(runner_train, runner_valid,
               step_op, train_op, summary_op, is_training_op,
               saved_vars, save_path, config.log_dir,
               network, network.restore)


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    main()
