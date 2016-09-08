import os
import pprint
import tensorflow as tf

from data_util import read_data
from models.test_model import TestModel

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("edim", 150, "internal state dimension [150]")
flags.DEFINE_integer("lindim", 75, "linear part of the state [75]")
flags.DEFINE_integer("batch_size", 128, "batch size to use during training [128]")
flags.DEFINE_integer("num_epoch", 100, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")
flags.DEFINE_string("data_dir", "data", "data directory [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_string("data_name", "en", "data set name [en]")
flags.DEFINE_string("target_name", "fr", "target data set name [fr]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
flags.DEFINE_boolean("show", False, "print progress [False]")
FLAGS = flags.FLAGS

train_data = "train_data"
valid_data = "valid_data"
test_data = "test_data"

def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:
        model = TestModel(sess, FLAGS)
        # model.build_model()

        if FLAGS.is_test:
            model.run(valid_data, test_data)
        else:
            model.run(train_data, valid_data)

if __name__ == '__main__':
    tf.app.run()
