from __future__ import division
import functools
from inspect import signature, Parameter
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

# from ops import *
# from utils import *

def auto_assign(func):
    # Signature:
    sig = signature(func)
    for name, param in sig.parameters.items():
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            raise RuntimeError('Unable to auto assign if *args or **kwargs in signature.')
    # Wrapper:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        for i, (name, param) in enumerate(sig.parameters.items()):
            # Skip 'self' param:
            if i == 0: continue
            # Search value in args, kwargs or defaults:
            if i - 1 < len(args):
                val = args[i - 1]
            elif name in kwargs:
                val = kwargs[name]
            else:
                val = param.default
            setattr(self, name, val)
        func(self, *args, **kwargs)
    return wrapper

class Model(object):
    @auto_assign
    def __init__(self, sess, config):
        """
        initialize
        """
        print("model initialize")

    def train(self, data):
        print("train", data)

    def test(self, data):
        print("test", data)

    def run(self, train_data, test_data):
        print("run")
        self.train(train_data)
        self.test(test_data)

    def save(self):
        print("save")
        # model_name = "DCGAN.model"
        # model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        # if not os.path.exists(checkpoint_dir):
            # os.makedirs(checkpoint_dir)

        # self.saver.save(self.sess,
                        # os.path.join(checkpoint_dir, model_name),
                        # global_step=step)

    def load(self):
        print(" [*] Reading checkpoints...")

        # model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # if ckpt and ckpt.model_checkpoint_path:
            # ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            # self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            # return True
        # else:
            # return False
