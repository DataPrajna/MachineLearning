# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from iq_data_preparation import IQData

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 64
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 5
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 1000
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 400  # Number of steps between evaluations.

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS
FLAGS.train_dir = "/home/ppdash/trained_model/iqa/"
FLAGS.train = True


class DQ:
    def __init__(self):
        self.prepare_data()
        self.model()
        #print(self.optimizer)
        #self.train()

    def error_rate(self, predictions, labels):
        """Return the error rate based on dense predictions and sparse labels."""
        return 100.0 - (
            100.0 *
            numpy.sum(numpy.argmax(predictions, 1) == labels) /
            predictions.shape[0])

    def prepare_data(self):
        a = IQData("../csiq.csv")
        d, l = a.get_samples()
        train_data = d
        train_labels = l
        test_data = d[1000:15000, :, :, :];
        test_labels = l[1000:15000]

        # Generate a validation set.
        self.validation_data = train_data[:VALIDATION_SIZE, ...]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, ...]
        self.train_labels = train_labels[VALIDATION_SIZE:]
        self.num_epochs = NUM_EPOCHS
        self.train_size = train_labels.shape[0]

    def model(self):
        self.train_data_node = tf.placeholder(
            tf.float32,
            shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        self.train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
        self.eval_data = tf.placeholder(
            tf.float32,
            shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

        # The variables below hold all the trainable weights. They are passed an
        # initial value which will be assigned when when we call:
        # {tf.initialize_all_variables().run()}
        conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                                stddev=0.1,
                                seed=SEED))
        conv1_biases = tf.Variable(tf.zeros([32]))

        conv2_weights = tf.Variable(
            tf.truncated_normal([3, 3, 32, 64],
                                stddev=0.1,
                                seed=SEED))
        conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

        conv3_weights = tf.Variable(
            tf.truncated_normal([3, 3, 64, 64],
                                stddev=0.1,
                                seed=SEED))
        conv3_biases = tf.Variable(tf.constant(0.1, shape=[64]))

        fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal(
                [IMAGE_SIZE // 8 * IMAGE_SIZE // 8 * 64, 512],
                stddev=0.1,
                seed=SEED))
        fc1_biases = tf.Variable(tf.constant(0.01, shape=[512]))
        fc2_weights = tf.Variable(
            tf.truncated_normal([512, NUM_LABELS],
                                stddev=0.1,
                                seed=SEED))
        fc2_biases = tf.Variable(tf.constant(0.01, shape=[NUM_LABELS]))

        conv = tf.nn.conv2d(self.train_data_node,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv = tf.nn.conv2d(pool,
                            conv3_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if FLAGS.train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)


        # Training computation: logits + cross-entropy loss.
        logits = tf.matmul(hidden, fc2_weights) + fc2_biases
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, self.train_labels_node))

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                        tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
        # Add the regularization term to the loss.
        loss += 5e-4 * regularizers

        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            0.01,  # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            self.train_size,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        self.optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                    0.9).minimize(loss,
                                                                  global_step=batch)
        print("working")
        print(self.optimizer)

        # Predictions for the current training minibatch.
        self.train_prediction = tf.nn.softmax(logits)

        # Predictions for the test and validation, which we'll compute less often.
        self.eval_prediction = tf.nn.softmax(model(eval_data))

    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(self, data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    self.eval_prediction,
                    feed_dict={self.eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    self.eval_prediction,
                    feed_dict={self.eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    def train(self):
        start_time = time.time()
        with tf.Session() as sess:
            # Run all the initializers to prepare the trainable parameters.
            tf.initialize_all_variables().run()
            print('Initialized!')
            # Loop through training steps.
            for step in xrange(int(self.num_epochs * self.train_size) // BATCH_SIZE):
                # Compute the offset of the current minibatch in the data.
                # Note that we could use better randomization across epochs.
                offset = (step * BATCH_SIZE) % (self.train_size - BATCH_SIZE)
                batch_data = self.train_data[offset:(offset + BATCH_SIZE), ...]
                batch_labels = self.train_labels[offset:(offset + BATCH_SIZE)]
                # This dictionary maps the batch data (as a numpy array) to the
                # node in the graph it should be fed to.train_data
                feed_dict = {self.train_data_node: batch_data,
                             self.train_labels_node: batch_labels}
                # Run the graph and fetch some of the nodes.
                _, l, lr, predictions = sess.run(
                    [self.optimizer, self.loss, self.learning_rate, self.train_prediction],
                    feed_dict=feed_dict)
                if step % EVAL_FREQUENCY == 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print('Step %d (epoch %.2f), %.1f ms' %
                          (step, float(step) * BATCH_SIZE / self.train_size,
                           1000 * elapsed_time / EVAL_FREQUENCY))
                    print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                    print('Minibatch error: %.1f%%' % self.error_rate(predictions, batch_labels))
                    print('Validation error: %.1f%%' % self.error_rate(
                        self.eval_in_batches(self.validation_data, sess), self.validation_labels))
                    sys.stdout.flush()
            # Finally print the result!
            test_error = self.error_rate(self.eval_in_batches(self.test_data, sess), self.test_labels)
            print('Test error: %.1f%%' % test_error)
            if FLAGS.self_test:
                print('test_error', test_error)
                assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
                    test_error,)




    def save_session(self, sess, step):
        saver = tf.train.Saver()
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    dq = DQ()
    #dq.prepare_data()
    #dq.train()
    #tf.app.run()
