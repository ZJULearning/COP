import collections
import tensorflow as tf
import numpy as np

from .classification_base import ClassificationBase

class VGG11(ClassificationBase):
    def get_params_and_calculation_from_channel_num(self, channel_num, num_classes, ori_size):
        """
        example: len([64, 128, 256, 256, 512, 512, 512, 512, 4096, 4096]) = 10
        """
        def get_input_size(index):
            size = ori_size
            if not isinstance(size, int):
                size = size[0]
            if index == 0:
                return size
            elif index == 1:
                return size / 2
            elif index >= 2 and index <= 3:
                return size / 4
            elif index >= 4 and index <= 5:
                return size / 8
            elif index >= 6 and index <= 7:
                return size / 16
            elif index >= 8:
                return 1
            return size
        def get_kernel_size(index):
            if index >= 8:
                return 1
            else:
                return 3
        ## params
        params = 0
        for i, output_channel in enumerate(channel_num):
            input_channel = 3 if i == 0 else channel_num[i-1]
            kernel_size = get_kernel_size(i)
            params += kernel_size * kernel_size * input_channel * output_channel + output_channel
        params += channel_num[-1] * num_classes + num_classes

        ## calculation
        calculation = 0
        for i, output_channel in enumerate(channel_num):
            input_channel = 3 if i == 0 else channel_num[i-1]
            input_size = get_input_size(i)
            kernel_size = get_kernel_size(i)
            calculation += 2 * input_size ** 2 * input_channel * (output_channel * kernel_size * kernel_size)
        calculation += 2 * channel_num[-1] * num_classes
        print("params: ", params, " calculation: ", calculation)
        return params, calculation

    def get_weights_from_model(self, model_path):
        """
        return weights_dict
        """
        reader = tf.train.NewCheckpointReader(model_path)
        all_variables = reader.get_variable_to_shape_map()
        print(all_variables.keys())
        kernel_weights = collections.OrderedDict()
        for i in range(1, 3):
            for j in range(1, 2):
                kernel_weights["conv%d/conv_%d/kernel" % (i, j)] = reader.get_tensor("vgg_11/conv%d/conv_%d/conv2d/kernel" % (i, j))
                kernel_weights["conv%d/conv_%d/bias" % (i, j)] = reader.get_tensor("vgg_11/conv%d/conv_%d/conv2d/bias" % (i, j))
        for i in range(3, 6):
            for j in range(1, 3):
                kernel_weights["conv%d/conv_%d/kernel" % (i, j)] = reader.get_tensor("vgg_11/conv%d/conv_%d/conv2d/kernel" % (i, j))
                kernel_weights["conv%d/conv_%d/bias" % (i, j)] = reader.get_tensor("vgg_11/conv%d/conv_%d/conv2d/bias" % (i, j))
        for i in range(1, 4):
            kernel_weights["dense%d/kernel" % i] = reader.get_tensor("vgg_11/dense%d/conv2d/kernel" % i)
            kernel_weights["dense%d/bias" % i] = reader.get_tensor("vgg_11/dense%d/conv2d/bias" % i)
        return kernel_weights

    def restore_weights(self, scope, layer_type, weights_dict):
        """
        prefix: scope[7:]
        layer_type: conv, bn, dense
        """
        if layer_type == "conv" or layer_type == "dense":
            saved_kernel = weights_dict.get(scope.name[7:] + "/kernel")
            saved_bias = weights_dict.get(scope.name[7:] + "/bias")
            if saved_kernel is not None:
                weight = tf.get_default_graph().get_tensor_by_name(scope.name + "/conv2d/kernel:0")
                weight = tf.assign(weight, saved_kernel)
                tf.add_to_collection("init", weight) # important
            if saved_bias is not None:
                bias = tf.get_default_graph().get_tensor_by_name(scope.name + "/conv2d/bias:0")
                bias = tf.assign(bias, saved_bias)
                tf.add_to_collection("init", bias)
        else:
            raise ValueError("unknown layer type")
        return

    def combine_layer(self, inputs, channels, name, use_bias, regularizer, weight_decay, 
        weights_dict={}, kernel_size=[3,3], strides=(1,1), padding="same"):
        with tf.variable_scope(name) as scope:
            net = tf.layers.conv2d(inputs, channels, kernel_size, strides=strides, padding=padding,
                kernel_regularizer=regularizer(weight_decay), use_bias=use_bias)
            self.restore_weights(scope, "conv", weights_dict)
            net = tf.nn.relu(net)
        return net

    def network(self, inputs, num_classes, scope, is_training, kargs):
        print("Use VGG-A")

        weights_dict = kargs.get("weights_dict") or {}
        weight_decay = kargs.weight_decay
        ori_channels_num = kargs.ori_channels_num
        feature_res = [] if kargs.get("get_features") else None
        regularizer = kargs.regularizer
        use_bias = kargs.use_bias
        print("Use weight_decay: ", weight_decay)

        # import IPython
        # IPython.embed()
        channel_num = kargs.channels_num
        if isinstance(channel_num, dict):
            channel_num = list(channel_num.values())
            print("Use set channels: ", channel_num)
        else:
            print("Use ori channels: ", channel_num)

        with tf.variable_scope(scope + "/conv1") as nsc:
            net = self.combine_layer(inputs, channel_num[0], "conv_1", use_bias, regularizer, weight_decay, weights_dict=weights_dict)
            net = tf.layers.max_pooling2d(net, 2, 2, name="max_pool", padding="valid")
        with tf.variable_scope(scope + "/conv2") as nsc:
            net = self.combine_layer(net, channel_num[1], "conv_1", use_bias, regularizer, weight_decay, weights_dict=weights_dict)
            net = tf.layers.max_pooling2d(net, 2, 2, name="max_pool", padding="valid")
        with tf.variable_scope(scope + "/conv3") as nsc:
            net = self.combine_layer(net, channel_num[2], "conv_1", use_bias, regularizer, weight_decay, weights_dict=weights_dict)
            net = self.combine_layer(net, channel_num[3], "conv_2", use_bias, regularizer, weight_decay, weights_dict=weights_dict)
            net = tf.layers.max_pooling2d(net, 2, 2, name="max_pool", padding="valid")
        with tf.variable_scope(scope + "/conv4") as nsc:
            net = self.combine_layer(net, channel_num[4], "conv_1", use_bias, regularizer, weight_decay, weights_dict=weights_dict)
            net = self.combine_layer(net, channel_num[5], "conv_2", use_bias, regularizer, weight_decay, weights_dict=weights_dict)
            net = tf.layers.max_pooling2d(net, 2, 2, name="max_pool", padding="valid")
        with tf.variable_scope(scope + "/conv5") as nsc:
            net = self.combine_layer(net, channel_num[6], "conv_1", use_bias, regularizer, weight_decay, weights_dict=weights_dict)
            net = self.combine_layer(net, channel_num[7], "conv_2", use_bias, regularizer, weight_decay, weights_dict=weights_dict)
            net = tf.layers.max_pooling2d(net, 2, 2, name="max_pool", padding="valid")
        
        with tf.variable_scope(scope + "/dense1") as nsc:
            net = tf.layers.conv2d(net, channel_num[8], [7,7], padding="valid", 
                kernel_regularizer=regularizer(weight_decay), use_bias=use_bias)
            self.restore_weights(nsc, "dense", weights_dict)
            net = tf.nn.relu(net, name="relu")
        with tf.variable_scope(scope + "/dense2") as nsc:
            net = tf.layers.dropout(net, rate=0.5, name="dropout1", training=is_training)
            net = tf.layers.conv2d(net, channel_num[9], [1, 1], kernel_regularizer=regularizer(weight_decay), use_bias=use_bias)
            self.restore_weights(nsc, "dense", weights_dict)
            net = tf.nn.relu(net, name="relu")
        with tf.variable_scope(scope + "/dense3") as nsc:
            net = tf.layers.dropout(net, rate=0.5, name="dropout1", training=is_training)
            net = tf.layers.conv2d(net, num_classes, [1, 1], kernel_regularizer=regularizer(weight_decay), use_bias=use_bias)
            self.restore_weights(nsc, "dense", weights_dict)
            net = tf.squeeze(net, [1, 2], name='squeezed')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.constant(0., dtype=tf.float32))
        return net