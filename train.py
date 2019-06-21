import numpy as np
import tensorflow as tf

from train_common import get_global_step, get_lr_and_max_steps, get_ops, run_op

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

tf.app.flags.DEFINE_string('train_dir', '/tmp/tmp_train', """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', './data', """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_string('dataset', 'cifar10', """dataset_name""")
tf.app.flags.DEFINE_string('network', 'vgg', """network_name""")

import config

network, dataset, top_name, _ = config.parse_net_and_dataset()

def train(_):
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

    store_model_path = FLAGS.train_dir
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        ## get global step, restoring the global step if an old checkpoint exists, or return 0
        global_step = get_global_step(store_model_path)

        ## learning rate
        lr, max_steps = get_lr_and_max_steps(examples_for_train, train_batch_size, num_gpus, 
            lr_decay_factor, epochs_per_decay, init_lr, global_step, lr_staircase, max_epochs)

        ## get optimizer
        opt = train_args.optimizer(lr)

        ## Get data
        tf_training = tf.placeholder(tf.bool, shape=())
        # train_dataset = dataset.distorted_inputs(256, "resnet18", use_std=True).make_one_shot_iterator()
        train_dataset = dataset.train_input_fn(FLAGS.data_dir, train_batch_size, max_epochs, **train_args.data_augmentation_args).make_one_shot_iterator()
        test_dataset = dataset.test_input_fn(FLAGS.data_dir, test_batch_size, **train_args.data_augmentation_args).make_one_shot_iterator()
        # test_dataset = dataset.inputs(False, 256, "resnet18", use_std=True).make_one_shot_iterator()

        ## get parameters and computational cost
        params, calculation = network.get_params_and_calculation_from_channel_num(train_args.ori_channels_num, num_classes, image_size)
        print("parameters: ", params, ", computation: ", calculation)

        ## put op on different GPU
        train_args.data_queue = [train_dataset, test_dataset]
        train_args.learning_rate = lr
        train_args.global_step = global_step
        train_args.max_steps = max_steps
        train_args.channels_num = train_args.ori_channels_num
        train_args.examples_per_epoch_for_test = examples_for_test

        ops = get_ops(opt, tf_training, network, dataset, num_classes, top_name, train_args)

        ###### run on session
        run_op(ops, tf_training, store_model_path, train_args)


if __name__ == '__main__':
    tf.app.run(main=train)