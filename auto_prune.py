from datetime import datetime
import os.path
import functools
import collections
import copy
import re
import time
import logging

import numpy as np
import tensorflow as tf

from train_common import get_global_step, get_lr_and_max_steps, get_ops, run_op

import prune_algorithm.prune_common as pc

logging.basicConfig(level=logging.ERROR)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/tmp_train', """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', './data', """Path to the data directory.""")
tf.app.flags.DEFINE_string('dataset', 'cifar10', """Dataset name""")
tf.app.flags.DEFINE_string('network', 'vgg', """Network name""")
tf.app.flags.DEFINE_string('impt_method', 'correlation', """should be one of `correlation`, `cosine` and `inner_product`""")
tf.app.flags.DEFINE_string('normalize_method', 'max', """should be one of `max`, `l1` and `l2`""")
tf.app.flags.DEFINE_bool('conv_dense_separate', False, """whether pruning conv and dense layers separately""")
tf.app.flags.DEFINE_bool('merge_all', False, """only for networks with residual design, whether mean the importance within the same block""")
tf.app.flags.DEFINE_float('prune_rate', 0.01, """The global pruned ratio for network""")
tf.app.flags.DEFINE_integer('top_k', 3, """The global pruned ratio for network""")
tf.app.flags.DEFINE_bool('weight_decay_growing', False, """Whether use a larger weight_decay when finetuning than training""")
tf.app.flags.DEFINE_float('alpha', 1.0, """The weight of 'correlation' when calculating the importance""")
tf.app.flags.DEFINE_float('beta', 1.0, """The weight of 'computational cost' when calculating the importance""")
tf.app.flags.DEFINE_float('gamma', 1.0, """The weight of 'parameters' when calculating the importance""")

import config
# import prune_config

network, dataset, top_name, PruneAlg = config.parse_net_and_dataset()

def train_with_graph(weights_dict, channel_num_after_pruned, weight_decay, store_model_path):
    train_args = config.args
    num_gpus = train_args.num_gpus
    train_batch_size = train_args.train_batch_size
    test_batch_size = train_args.test_batch_size
    init_lr = train_args.initial_learning_rate
    epochs_per_decay = train_args.num_epochs_per_decay
    lr_decay_factor = train_args.learning_rate_decay_factor
    lr_staircase = train_args.get("staircase")
    max_epochs = train_args.max_epochs

    num_classes = dataset.num_classes
    image_size = [dataset.height, dataset.width]
    examples_for_train = dataset.num_examples_for_train
    examples_for_test = dataset.num_examples_for_test
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        ###### get global step
        global_step = get_global_step(store_model_path)

        ###### get learning rate and max_steps
        lr, max_steps = get_lr_and_max_steps(examples_for_train, train_batch_size, num_gpus,
            lr_decay_factor, epochs_per_decay, init_lr, global_step, lr_staircase, max_epochs)

        ###### get optimizer
        opt = config.args.optimizer(lr)

        ###### Get data
        tf_training = tf.placeholder(tf.bool, shape=())
        train_dataset = dataset.train_input_fn(FLAGS.data_dir, train_batch_size, max_epochs, **config.args.data_augmentation_args).make_one_shot_iterator()
        test_dataset = dataset.test_input_fn(FLAGS.data_dir, test_batch_size).make_one_shot_iterator()

        ###### put op on different GPU
        train_args.learning_rate = lr
        train_args.data_queue = [train_dataset, test_dataset]
        train_args.global_step = global_step
        train_args.max_steps = max_steps
        train_args.examples_per_epoch_for_test = examples_for_test
        train_args.weights_dict = weights_dict
        train_args.channels_num = channel_num_after_pruned
        train_args.weight_decay = weight_decay

        ops = get_ops(opt, tf_training, network, dataset, num_classes, top_name, train_args)

        ###### run on session
        run_op(ops, tf_training, store_model_path, train_args)


def train(_):
    corr_normal_factor = collections.OrderedDict()
    ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.train_dir))
    store_model_path = ckpt.model_checkpoint_path
    init_weight_decay = config.args.weight_decay

    importance_coefficient = [FLAGS.alpha, FLAGS.beta, FLAGS.gamma]

    print("store_model_path: " + store_model_path)
    print("importance_coefficient: alpha %.2f, beta %.2f, gamma %.2f" % (FLAGS.alpha, FLAGS.beta, FLAGS.gamma))
    print("weight decay growing: %d" % FLAGS.weight_decay_growing)
    print("top_k: %d" % FLAGS.top_k)
    print("prune rate: %.2f" % FLAGS.prune_rate)

    ## get old weights
    if "resnet" in top_name:
        weights_dict = network.get_weights_from_model(store_model_path, config.args.resnet_version)
    else:
        weights_dict = network.get_weights_from_model(store_model_path)
    ## get pruned channels
    prune_args = {
        "image_size": [dataset.height, dataset.width],
        "importance_method": FLAGS.impt_method,
        "importance_coefficient": [FLAGS.alpha, FLAGS.beta, FLAGS.gamma],
        "top_k": FLAGS.top_k,
        "num_classes": dataset.num_classes,
        "normalize_method": FLAGS.normalize_method,
        "conv_dense_separate": False if FLAGS.conv_dense_separate == 0 else True,
        "merge_all": FLAGS.merge_all
    }
    prune_alg = PruneAlg(weights_dict, **prune_args)

    cut_channels = prune_alg.get_prune_channels(FLAGS.prune_rate)

    ## get pruned weights
    if "resnet" in top_name:
        pruned_weights_dict = prune_alg.get_pruned_weights(cut_channels, config.args.resnet_version)
    else:
        pruned_weights_dict = prune_alg.get_pruned_weights(cut_channels)
    cal_ratio, params_ratio = prune_alg.get_pruned_ratio()

    pruned_cared_weights = prune_alg.get_pruned_cared_weights(pruned_weights_dict)
    channel_num_after_pruned = prune_alg.get_channels_nums(pruned_cared_weights, channel_type='output')
    ## cal weight_decay
    weight_decay = 1.1e-3 if FLAGS.weight_decay_growing else init_weight_decay

    print("The number of channels after pruned: ", channel_num_after_pruned.values())
    print("Use correlation normalization factor: " + str(corr_normal_factor.values()))
    print("Use weight decay: " + str(weight_decay))

    ## finetune the model
    store_model_path = os.path.join(FLAGS.train_dir, "prune%.2f" % FLAGS.prune_rate) # model dir
    train_with_graph(pruned_weights_dict, channel_num_after_pruned, weight_decay, store_model_path)

if __name__ == '__main__':
    tf.app.run(main=train)