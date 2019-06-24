import copy
import tensorflow as tf

import networks.vgg16 as vgg16
import networks.vgg11 as vgg11
import networks.mobilenet_for_cifar as mobilenet_for_cifar
import networks.mobilenet_for_imagenet as mobilenet_for_imagenet
import networks.resnet18 as resnet18
import networks.resnet32 as resnet32

import datasets.cifar100 as cifar100
import datasets.cifar10 as cifar10
import datasets.cifar10 as cifar10
import datasets.cifar100 as cifar100
import datasets.imagenet as imagenet

import prune_algorithm.prune_vgg11 as prune_vgg11
import prune_algorithm.prune_vgg16 as prune_vgg16
import prune_algorithm.prune_mobilenet_for_cifar as prune_mobilenet_for_cifar
import prune_algorithm.prune_mobilenet_for_imagenet as prune_mobilenet_for_imagenet
import prune_algorithm.prune_resnet32 as prune_resnet32
import prune_algorithm.prune_resnet18 as prune_resnet18

FLAGS = tf.app.flags.FLAGS

## network and dataset
dataset_name = FLAGS.dataset # "cifar10" "cifar100", "imagenet"
network_name = FLAGS.network # "vgg11" "vgg16" "mobilenet_for_cifar" "mobilenet_for_imagenet" "resnet32"

class TrainArgs(object):
    def get(self, attr_name):
        try:
            res = getattr(self, attr_name)
        except AttributeError as e:
            res = None
        return res

args = TrainArgs()

def parse_net_and_dataset():
    if network_name == "vgg16":
        network = vgg16.VGG16()
        scope = "vgg_16"
        # prune_alg = channel_wise_corr_vgg16
        prune_alg = prune_vgg16.PruneVgg16
    elif network_name == "vgg11":
        network = vgg11.VGG11()
        scope = "vgg_11"
        prune_alg = prune_vgg11.PruneVgg11
    elif network_name == "mobilenet_for_cifar":
        network = mobilenet_for_cifar.MobileNetForCifar()
        scope = "mobilenet_for_cifar"
        prune_alg = prune_mobilenet_for_cifar.PruneMobileNetForCifar
    elif network_name == "mobilenet_for_imagenet":
        network = mobilenet_for_imagenet.MobileNetForImagenet()
        scope = "mobilenet_for_imagenet"
        prune_alg = prune_mobilenet_for_imagenet.PruneMobileNetForImagenet
    elif network_name == "resnet32":
        network = resnet32.ResNet32()
        scope = "resnet32"
        prune_alg = prune_resnet32.PruneResNet32
    elif network_name == "resnet18":
        network = resnet18.ResNet18()
        scope = "resnet18"
        prune_alg = prune_resnet18.PruneResNet18
    else:
        raise ValueError("unknown network name")

    if dataset_name == "cifar100":
        dataset = cifar100
    elif dataset_name == "cifar10":
        dataset = cifar10
    elif dataset_name == "imagenet":
        dataset = imagenet
    else:
        raise ValueError("unknown dataset name")

    return network, dataset, scope, prune_alg 


## parameters for network
if network_name == "vgg11":
    if dataset_name == "imagenet":
        args.train_batch_size = 128
        args.test_batch_size = 100
        args.image_size = [224, 224]
        args.num_gpus = 2
        args.use_bn = False
        args.use_bias = True
        args.weight_decay = 5e-4
        args.staircase = False
        args.regularizer = tf.contrib.slim.l2_regularizer
        args.optimizer = lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        args.num_epochs_per_decay = [30, 60, 90]
        args.learning_rate_decay_factor = [1, 0.1, 0.01, 0.001]
        args.initial_learning_rate = 0.01
        args.max_epochs = 120
        # args.channels_num = [64, 128, 256, 256, 512, 512, 512, 512, 4096, 4096]
        args.ori_channels_num = [64, 128, 256, 256, 512, 512, 512, 512, 4096, 4096]
        args.data_augmentation_args = {"resize": True, "crop_bbox": False, "padding": False,
                                       "bright": False, "mirroring": True, 
                                       "mean": [123.68, 116.779, 103.939], "std": [1.0, 1.0, 1.0]}

elif network_name == "vgg16":
    if "cifar" in dataset_name:
        args.train_batch_size = 128
        args.test_batch_size = 100
        args.image_size = [32, 32]
        args.num_gpus = 1
        args.use_bn = True
        args.use_bias = True
        args.weight_decay = 1.5e-3
        args.staircase = False
        args.regularizer = tf.contrib.slim.l2_regularizer
        args.optimizer = lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        args.num_epochs_per_decay = 20
        args.learning_rate_decay_factor = 0.5
        args.initial_learning_rate = 0.1
        args.max_epochs = 250
        # args.channels_num = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512]
        args.ori_channels_num = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512]
        args.data_augmentation_args = {"padding": True, "bright": False, "mirroring": True, 
                                        "mean": 120.707, "std": 64.15}
        
        if dataset_name == "cifar10":
            args.init_dropout = [0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5]
        elif dataset_name == "cifar100":
            args.init_dropout = [0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4]
    else:
        raise ValueError("unknown dataset name")


elif network_name == "mobilenet_for_cifar":
    if "cifar" in dataset_name:
        args.train_batch_size = 64
        args.test_batch_size = 100
        args.image_size = [32, 32]
        args.num_gpus = 1
        args.use_bn = True
        args.use_bias = False
        # channels_num = [28, 45, 81, 79, 221, 188, 395, 406, 319, 382, 329, 498, 691, 687] # 0.67
        # channels_num = [24, 48, 96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 726, 726] # 0.75
        # channels_num = [26, 39, 64, 63, 167, 144, 282, 239, 189, 254, 153, 482, 418, 390] # 0.38
        # channels_num = [16, 32, 64, 64, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512] # 0.50
        # channels_num = [8, 16, 32, 32, 64, 64 ,128, 128, 128, 128, 128, 128, 256, 256] # 0.25
        # args.channels_num = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        args.ori_channels_num = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        args.weight_decay = 6e-4
        args.staircase = False
        args.optimizer = lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        args.initializer = tf.contrib.layers.variance_scaling_initializer
        args.regularizer = tf.contrib.slim.l2_regularizer # depth-wise convolution do not use regularizer
        args.num_epochs_per_decay = 20
        args.learning_rate_decay_factor = 0.5
        args.initial_learning_rate = 0.1
        args.max_epochs = 125
        args.data_augmentation_args = {"padding": True, "bright": True, "mirroring": True,
                                       "mean": 120.707, "std": 64.15}

elif network_name == "mobilenet_for_imagenet":
    if dataset_name == "imagenet":
        batch_size_base = 256
        args.train_batch_size = 128
        args.test_batch_size = 100
        args.image_size = [224, 224]
        args.num_gpus = 2
        args.use_bn = True
        args.use_bias = True
        # channels_num = [28, 45, 81, 79, 221, 188, 395, 406, 319, 382, 329, 498, 691, 687] # 0.67
        # channels_num = [24, 48, 96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 726, 726] # 0.75
        # channels_num = [26, 39, 64, 63, 167, 144, 282, 239, 189, 254, 153, 482, 418, 390] # 0.38
        # channels_num = [16, 32, 64, 64, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512] # 0.50
        # channels_num = [8, 16, 32, 32, 64, 64 ,128, 128, 128, 128, 128, 128, 256, 256] # 0.25
        # args.channels_num = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        args.ori_channels_num = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        args.weight_decay = 4e-5
        args.use_nesterov = True
        args.optimizer = lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=False)
        args.initializer = lambda: tf.truncated_normal_initializer(stddev=0.09)
        args.regularizer = tf.contrib.slim.l2_regularizer # depth-wise convolution do not use regularizer
        args.num_epochs_per_decay = [30, 60, 90]
        args.learning_rate_decay_factor = [1, 0.1, 0.01, 0.001]
        args.initial_learning_rate = 0.1 * (args.train_batch_size * args.num_gpus / batch_size_base)
        args.max_epochs = 120
        args.data_augmentation_args = {"crop_bbox": True, "padding": False, "resize": False,
                                       "bright": True, "mirroring": True,
                                       "mean": 127.5, "std": 127.5}

elif network_name == "resnet32":
    if "cifar" in dataset_name:
        args.train_batch_size = 128
        args.test_batch_size = 100
        args.image_size = [32, 32]
        args.num_gpus = 1
        args.use_bias = False
        args.weight_decay = 2e-4
        args.staircase = True
        args.optimizer = lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=False)
        args.initializer = tf.contrib.layers.variance_scaling_initializer
        args.regularizer = tf.contrib.slim.l2_regularizer
        args.num_epochs_per_decay = [100, 150, 200]
        args.learning_rate_decay_factor = [1, 0.1, 0.01, 0.001]
        args.initial_learning_rate = 0.1
        args.max_epochs = 250
        args.data_augmentation_args = {"padding": True, "bright": False, "mirroring": True,
                                       "mean": 120.707, "std": 64.15}

        args.block_sizes = [5, 5, 5]
        # channels_num = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        #                     32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        #                     64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        # channels_num = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 
        #                     25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 
        #                     51, 51, 51, 51, 51, 51, 51, 51, 51, 51]
        # args.channels_num = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        #                     32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        #                     64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        # channels_num = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        #                     20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        #                     40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
        args.ori_channels_num = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                                32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                                64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        args.strides = [1, 2, 2]
        args.resnet_version = 2 # 1 or 2

        args.init_dropout = []

elif network_name == "resnet18":
    if dataset_name == "imagenet":
        batch_size_base = 256
        args.train_batch_size = 256
        args.test_batch_size = 100
        args.num_gpus = 2
        args.use_bias = False
        args.weight_decay = 1e-4
        args.initializer = lambda: tf.contrib.layers.variance_scaling_initializer()
        args.regularizer = tf.contrib.slim.l2_regularizer
        args.num_epochs_per_decay = 30
        args.initial_learning_rate = 0.1 * args.num_gpus * args.train_batch_size / batch_size_base
        args.learning_rate_decay_factor = 0.1
        args.max_epochs = 110
        args.optimizer = lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        args.staircase = True
        args.data_augmentation_args = {"crop_bbox": True, "padding": False, "resize": False,
                                       "bright": False, "mirroring": True,
                                       "mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375]}

        args.block_sizes = [2, 2, 2, 2]
        # init_channels = 64
        # channels_num = [48, 48, 48, 48, 48, 96, 96, 96, 96, 192, 192, 192, 192, 384, 384, 384, 384] # 0.75
        # channels_num = [49, 49, 49, 49, 49, 98, 98, 98, 98, 196, 196, 196, 196, 392, 392, 392, 392]
        # channels_num = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
        args.ori_channels_num = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
        args.strides = [1, 2, 2, 2]

