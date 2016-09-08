
"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

class Seq2Seq(object):
    def __init__(self, config, sess):
        self.nwords = config.nwords
        self.init_hid = config.init_hid # initialize internal state value
        self.init_std = config.init_std # weight initialization std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch # number of epoch to use during training
        self.edim = config.edim # internal state dimension

        self.show = config.show
        self.is_test = config.is_test
        self.checkpoint_dir = config.checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            raise Exception(" [!] Directory %s not found" % self.checkpoint_dir)

        self.input = tf.placeholder(tf.float32, [None, self.edim])
        self.target = tf.placeholder(tf.float32, [self.batch_size, self.nwords],
                                     name="target")

        self.hid = []
        self.hid.append(self.input)
        self.share_list = []
        self.share_list.append([])

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess

    def build_model(self):
        self.build_memory()

        self.W = tf.Variable(tf.random_normal([self.edim, self.nwords], stddev=self.init_std))
        z = tf.matmul(self.hid[-1], self.W)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(z, self.target)

        self.lr = tf.Variable(self.current_lr)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        params = [self.A, self.B, self.C, self.T_A, self.T_B, self.W]
        grads_and_vars = self.opt.compute_gradients(self.loss,params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                   for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()

    def train(self, data):
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])

        x.fill(self.init_hid)
        for t in xrange(self.mem_size):
            time[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('Train', max=N)

        for idx in xrange(N):
            if self.show: bar.next()
            target.fill(0)
            for b in xrange(self.batch_size):
                m = random.randrange(self.mem_size, len(data))
                target[b][data[m]] = 1
                context[b] = data[m - self.mem_size:m]

            _, loss, self.step = self.sess.run([self.optim,
                                                self.loss,
                                                self.global_step],
                                                feed_dict={
                                                    self.input: x,
                                                    self.time: time,
                                                    self.target: target,
                                                    self.context: context})
            cost += np.sum(loss)

        if self.show: bar.finish()
        return cost/N/self.batch_size

    def test(self, data, label="test"):
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])

        x.fill(self.init_hid)
        for t in xrange(self.mem_size):
            time[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar(label, max=N)

        m = self.mem_size
        for idx in xrange(N):
            if self.show: bar.next()
            target.fill(0)
            for b in xrange(self.batch_size):
                target[b][data[m]] = 1
                context[b] = data[m - self.mem_size:m]
                m += 1

                if m >= len(data):
                    m = self.mem_size

            loss = self.sess.run([self.loss], feed_dict={self.input: x,
                                                         self.time: time,
                                                         self.target: target,
                                                         self.context: context})
            cost += np.sum(loss)

        if self.show: bar.finish()
        return cost/N/self.batch_size

    def run(self, train_data, test_data):
        # Training
        if not self.is_test:
            for idx in xrange(self.nepoch):
                train_loss = np.sum(self.train(train_data))
                test_loss = np.sum(self.test(test_data), label="validation")

                state = {
                        "perplexity": math.exp(train_loss),
                        "epoch":idx,
                        "learning_rate": self.current_lr,
                        "valid_perplexity":math.exp(test_loss)
                        }
                print(state)

                if idx % 10 == 0:
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir,
                                                 "Seq2Seq3.model"),
                                    global_step = self.step.astype(int))
        # Testing
        else:
            self.load()
            valid_loss = np.sum(self.test(train_data), label="validation")
            test_loss = np.sum(self.test(test_data), label="test")

            state = {
                'valid_perplexity': math.exp(valid_loss),
                'test_perplexity': math.exp(test_loss)
            }
            print(state)

    def load(self):
        print(" [*] Reading Checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Test mode but no checkpoint found")
