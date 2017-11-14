import pickle

import tensorflow as tf
import numpy as np

import alexnet
import vgg19
import vgg

import config
import dataprovider
import multithread_generator
import ops

import resnet

slim = tf.contrib.slim


class NetRunner(object):
    def __init__(self, save_path, num_classes,
                 x_op=None, labels_op=None, is_training_op=None,
                 tree_path=None):
        # Attributes.
        self.save_path = save_path
        self.num_classes = num_classes

        if is_training_op is None:
            self.is_training_op = tf.placeholder(tf.bool)
        else:
            self.is_training_op = is_training_op

        self.eval_mode = (x_op is None)

        # Training variables.
        if not self.eval_mode:
            self.x_op = x_op
            self.labels_op = labels_op

            # Used for training.
            self.pred_op, self.models, _ = self._build(x_op, num_classes, is_training_op)
            self.loss_op, self.losses = self._make_loss(self.models, labels_op)

        # More hacks.
        self.sess = None
        self.input_op = None
        self.pred_op_eval = None
        self.prob_op_eval = None
        self.tree_path = tree_path
        self.tree = None

    def _build(self, input_op, num_classes, is_training_op, reuse=False):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            # vgg = vgg19.VGG19('imagenet-vgg-verydeep-19.mat').feed_forward(input_op)

            models = [
                    # Net.feed_forward(input_op, num_classes, is_training_op)
                    # ResNet.feed_forward_vgg(vgg, num_classes, is_training_op, 'conv4_1')
                    Net.resnet56(input_op, num_classes, is_training_op, 'Net')
                    # Net.feed_forward_alexnet(input_op, num_classes, is_training_op)
                    # Net.feed_forward_vgg16(input_op, num_classes, is_training_op)
                    ]

        # Average the predictions.
        pred = 0.0
        prob = list()

        for pred_op, _, prob_op in models:
            pred += tf.to_float(pred_op)
            prob.append(prob_op)

        prob = tf.stack(prob)
        pred = tf.cast(tf.round(pred / len(models)), tf.int32)

        return pred, models, prob

    def _make_loss(self, models, labels_op):
        loss_op = 0.0
        losses = list()

        for _, logits_op, _ in models:
            losses.append(
                    Net.get_loss_op(
                        logits_op, labels_op))
            loss_op += losses[-1]

        return loss_op, losses

    def restore(self, sess=None):
        if sess is None:
            sess = tf.Session()
            self.sess = sess

        saved_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Net')
        saved_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'training')

        saved_vars_dict = {key.name: key for key in saved_vars}
        saver = tf.train.Saver(saved_vars_dict)

        try:
            saver.restore(sess, self.save_path)
            print('Loaded weights.')
        except Exception as e:
            print(e)
            print('Reinitializing')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

    def predict(self, image):
        if self.pred_op_eval is None:
            self.input_op = tf.placeholder(
                    tf.float32, shape=[1]+config.input_shape)
            self.pred_op_eval, _, self.prob_op_eval = self._build(
                    self.input_op, self.num_classes, self.is_training_op,
                    not self.eval_mode)

        if self.sess is None:
            self.restore()

        pred, prob = self.sess.run(
                [self.pred_op_eval, self.prob_op_eval],
                {self.input_op: np.expand_dims(image, 0),
                 self.is_training_op: False})

        return pred, prob

class Net(object):
    def resnet56(x_op, num_classes, is_training_op, scope):
        with tf.variable_scope(scope):
            model_func = resnet.cifar10_resnet_v2_generator(56, num_classes)
            pred_op, net, prob_op = model_func(x_op, is_training_op)
            return pred_op, net, prob_op

    def get_loss_op(logits_op, labels_op, alpha=5e-7, beta=5.0):
        with tf.name_scope('loss'):
            cross_entropy_loss_op = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits_op, labels=labels_op))

            loss_op = cross_entropy_loss_op

        return loss_op

    def get_runners(input_shape, num_classes):
        datagen_train = dataprovider.SparseDatagen(
                True, input_shape, num_classes, augment_data=True)
        datagen_valid = dataprovider.SparseDatagen(
                False, input_shape, num_classes, augment_data=False)

        runner_train = multithread_generator.AsyncRunner(
                datagen_train, 'train_generator')
        runner_valid = multithread_generator.AsyncRunner(
                datagen_valid, 'valid_generator')

        return runner_train, runner_valid

    def get_summaries(images_op, labels_op, pred_op,
                      num_classes, learn_rate_op, loss_op,
                      grad_var_op, is_training_op, network):
        def get_summary(is_training):
            result = list()

            if is_training:
                result += tf.get_collection(tf.GraphKeys.SUMMARIES)
                result += [tf.summary.scalar('learn_rate', learn_rate_op)]

            with tf.name_scope('train' if is_training else 'valid'):
                result += [tf.summary.image('images', images_op, 10)]

                result += [tf.summary.scalar('accuracy',
                    tf.reduce_mean(tf.to_float(tf.equal(
                        pred_op, labels_op))))]

                result += [tf.summary.image(
                    'confusion_matrix',
                    ops.confusion_image(pred_op, labels_op, num_classes))]

            return tf.summary.merge(result)

        return tf.cond(
                is_training_op,
                lambda: get_summary(True),
                lambda: get_summary(False), name='')
