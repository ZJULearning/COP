import collections
import tensorflow as tf
import numpy as np

from .classification_base import ClassificationBase

class VGG16(ClassificationBase):
    def get_params_and_calculation_from_channel_num(self, channel_num, num_classes, ori_size):
        """
        example: len([64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512]) = 13
        """
        def get_input_size(index):
            size = ori_size
            if not isinstance(size, int):
                size = size[0]
            if index >= 0 and index <= 1:
                return size
            elif index >= 2 and index <= 3:
                return size / 2
            elif index >= 4 and index <= 6:
                return size / 4
            elif index >= 7 and index <= 9:
                return size / 8
            elif index >= 10 and index <= 12:
                return size / 16
            elif index >= 13:
                return size / 32
            return size
        def get_kernel_size(index):
            if index >= 13:
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
        # print("params: ", params, " calculation: ", calculation)
        return params, calculation

    def get_weights_from_model(self, model_path):
        """
        return weights_dict
        """
        reader = tf.train.NewCheckpointReader(model_path)
        all_variables = reader.get_variable_to_shape_map()
        # print(all_variables)
        kernel_weights = collections.OrderedDict()
        for i in range(1, 3):
            for j in range(1, 3):
                kernel_weights["conv%d/conv_%d/kernel" % (i, j)] = reader.get_tensor("vgg_16/conv%d/conv_%d/conv2d/kernel" % (i, j))
                kernel_weights["conv%d/conv_%d/bias" % (i, j)] = reader.get_tensor("vgg_16/conv%d/conv_%d/conv2d/bias" % (i, j))
                kernel_weights["conv%d/conv_%d/beta" % (i, j)] = reader.get_tensor("vgg_16/conv%d/conv_%d/batch_normalization/beta" % (i, j))
                kernel_weights["conv%d/conv_%d/gamma" % (i, j)] = reader.get_tensor("vgg_16/conv%d/conv_%d/batch_normalization/gamma" % (i, j))
                kernel_weights["conv%d/conv_%d/moving_mean" % (i, j)] = reader.get_tensor("vgg_16/conv%d/conv_%d/batch_normalization/moving_mean" % (i, j))
                kernel_weights["conv%d/conv_%d/moving_variance" % (i, j)] = reader.get_tensor("vgg_16/conv%d/conv_%d/batch_normalization/moving_variance" % (i, j))
        for i in range(3, 6):
            for j in range(1, 4):
                kernel_weights["conv%d/conv_%d/kernel" % (i, j)] = reader.get_tensor("vgg_16/conv%d/conv_%d/conv2d/kernel" % (i, j))
                kernel_weights["conv%d/conv_%d/bias" % (i, j)] = reader.get_tensor("vgg_16/conv%d/conv_%d/conv2d/bias" % (i, j))
                kernel_weights["conv%d/conv_%d/beta" % (i, j)] = reader.get_tensor("vgg_16/conv%d/conv_%d/batch_normalization/beta" % (i, j))
                kernel_weights["conv%d/conv_%d/gamma" % (i, j)] = reader.get_tensor("vgg_16/conv%d/conv_%d/batch_normalization/gamma" % (i, j))
                kernel_weights["conv%d/conv_%d/moving_mean" % (i, j)] = reader.get_tensor("vgg_16/conv%d/conv_%d/batch_normalization/moving_mean" % (i, j))
                kernel_weights["conv%d/conv_%d/moving_variance" % (i, j)] = reader.get_tensor("vgg_16/conv%d/conv_%d/batch_normalization/moving_variance" % (i, j))
        for i in range(1, 3):
            kernel_weights["dense%d/kernel" % i] = reader.get_tensor("vgg_16/dense%d/conv2d/kernel" % i)
            kernel_weights["dense%d/bias" % i] = reader.get_tensor("vgg_16/dense%d/conv2d/bias" % i)
            if i == 1:
                kernel_weights["dense%d/beta" % i] = reader.get_tensor("vgg_16/dense%d/batch_normalization/beta" % i)
                kernel_weights["dense%d/gamma" % i] = reader.get_tensor("vgg_16/dense%d/batch_normalization/gamma" % i)
                kernel_weights["dense%d/moving_mean" % i] = reader.get_tensor("vgg_16/dense%d/batch_normalization/moving_mean" % i)
                kernel_weights["dense%d/moving_variance" % i] = reader.get_tensor("vgg_16/dense%d/batch_normalization/moving_variance" % i)
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
        elif layer_type == "bn":
            saved_beta = weights_dict.get(scope.name[7:] + "/beta")
            saved_gamma = weights_dict.get(scope.name[7:] + "/gamma")
            saved_moving_mean = weights_dict.get(scope.name[7:] + "/moving_mean")
            saved_moving_variance = weights_dict.get(scope.name[7:] + "/moving_variance")
            if saved_beta is not None:
                beta = tf.get_default_graph().get_tensor_by_name(scope.name + "/batch_normalization/beta:0")
                gamma = tf.get_default_graph().get_tensor_by_name(scope.name + "/batch_normalization/gamma:0")
                moving_mean = tf.get_default_graph().get_tensor_by_name(scope.name + "/batch_normalization/moving_mean:0")
                moving_variance = tf.get_default_graph().get_tensor_by_name(scope.name + "/batch_normalization/moving_variance:0")

                beta = tf.assign(beta, saved_beta)
                gamma = tf.assign(gamma, saved_gamma)
                moving_mean = tf.assign(moving_mean, saved_moving_mean)
                moving_variance = tf.assign(moving_variance, saved_moving_variance)

                tf.add_to_collection("init", beta)
                tf.add_to_collection("init", gamma)
                tf.add_to_collection("init", moving_mean)
                tf.add_to_collection("init", moving_variance)
        else:
            raise ValueError("unknown layer type")
        return

    def combine_layer(self, inputs, channels, name, is_training, use_bias, drop_rate, regularizer, weight_decay,
        weights_dict={}, get_features=False, kernel_size=[3,3], strides=(1,1), padding="same"):
        with tf.variable_scope(name) as scope:
            net = tf.layers.conv2d(inputs, channels, kernel_size, strides=strides, padding=padding,
                kernel_regularizer=regularizer(weight_decay), use_bias=use_bias)
            self.restore_weights(scope, "conv", weights_dict)
            feature = net
            net = tf.nn.relu(net)

            # tensorflow batch normalization
            net = tf.layers.batch_normalization(net, axis=-1, training=is_training)
            self.restore_weights(scope, "bn", weights_dict)
            # dropout
            if drop_rate > 1e-3:
                net = tf.layers.dropout(net, rate=drop_rate, training=is_training)
        if get_features:
            return net, feature
        else:
            return net

    def get_dropout_rate(self, init_dropout, channels_num, ori_channels_num, classes):
        if init_dropout is None:
            return [0.0] * 10
        pick_layers_index = [0, 2, 4, 5, 7, 8, 10, 11, 12, 13]
        ori_pick_channels = [ori_channels_num[i] for i in pick_layers_index]
        ori_pick_channels_1 = [(ori_channels_num[i+1] if i + 1 < len(ori_channels_num) else classes) for i in pick_layers_index]
        now_pick_channels = [channels_num[i] for i in pick_layers_index]
        now_pick_channels_1 = [(channels_num[i+1] if i + 1 < len(channels_num) else classes) for i in pick_layers_index]

        dropout_rate = [dropout * \
            ((now_pick_channels[i] * now_pick_channels_1[i]) / (ori_pick_channels[i] * ori_pick_channels_1[i]))**0.5 \
            for i, dropout in enumerate(init_dropout)]

        return dropout_rate

    def network(self, inputs, num_classes, scope, is_training, kargs):
        print("Use VGG16")
        weights_dict = kargs.get("weights_dict") or {}
        weight_decay = kargs.weight_decay
        ori_channels_num = kargs.ori_channels_num
        feature_res = [] if kargs.get("get_features") else None
        init_dropout = kargs.get("init_dropout")
        regularizer = kargs.regularizer
        use_bias = kargs.use_bias
        print("Use weight_decay: ", weight_decay)

        channel_num = kargs.channels_num
        if isinstance(channel_num, dict):
            channel_num = list(channel_num.values())
            print("Use set channels: ", channel_num)
        else:
            print("Use ori channels: ", channel_num)
        dropout = self.get_dropout_rate(init_dropout, channel_num, ori_channels_num, num_classes)
        print("Use dropout: ", dropout)
        with tf.variable_scope(scope + "/conv1") as nsc:
            net, feature = self.combine_layer(inputs, channel_num[0], "conv_1", is_training, use_bias, dropout[0], regularizer, weight_decay, weights_dict, get_features=True)
            if feature_res is not None:
                feature_res.append(feature)
            net, feature = self.combine_layer(net, channel_num[1], "conv_2", is_training, use_bias, 0.0, regularizer, weight_decay, weights_dict, get_features=True)
            if feature_res is not None:
                feature_res.append(feature)
            net = tf.layers.max_pooling2d(net, 2, 2, name="max_pool", padding="valid")
        with tf.variable_scope(scope + "/conv2") as nsc:
            net, feature = self.combine_layer(net, channel_num[2], "conv_1", is_training, use_bias, dropout[1], regularizer, weight_decay, weights_dict, get_features=True)
            net = self.combine_layer(net, channel_num[3], "conv_2", is_training, use_bias, 0.0, regularizer, weight_decay, weights_dict)
            net = tf.layers.max_pooling2d(net, 2, 2, name="max_pool", padding="valid")
        with tf.variable_scope(scope + "/conv3") as nsc:
            net = self.combine_layer(net, channel_num[4], "conv_1", is_training, use_bias, dropout[2], regularizer, weight_decay, weights_dict)
            net = self.combine_layer(net, channel_num[5], "conv_2", is_training, use_bias, dropout[3], regularizer, weight_decay, weights_dict)
            net = self.combine_layer(net, channel_num[6], "conv_3", is_training, use_bias, 0.0, regularizer, weight_decay, weights_dict)
            net = tf.layers.max_pooling2d(net, 2, 2, name="max_pool", padding="valid")
        with tf.variable_scope(scope + "/conv4") as nsc:
            net = self.combine_layer(net, channel_num[7], "conv_1", is_training, use_bias, dropout[4], regularizer, weight_decay, weights_dict)
            net = self.combine_layer(net, channel_num[8], "conv_2", is_training, use_bias, dropout[5], regularizer, weight_decay, weights_dict)
            net = self.combine_layer(net, channel_num[9], "conv_3", is_training, use_bias, 0.0, regularizer, weight_decay, weights_dict)
            net = tf.layers.max_pooling2d(net, 2, 2, name="max_pool", padding="valid")
        with tf.variable_scope(scope + "/conv5") as nsc:
            net = self.combine_layer(net, channel_num[10], "conv_1", is_training, use_bias, dropout[6], regularizer, weight_decay, weights_dict)
            net = self.combine_layer(net, channel_num[11], "conv_2", is_training, use_bias, dropout[7], regularizer, weight_decay, weights_dict)
            net = self.combine_layer(net, channel_num[12], "conv_3", is_training, use_bias, 0.0, regularizer, weight_decay, weights_dict)
            net = tf.layers.max_pooling2d(net, 2, 2, name="max_pool", padding="valid")
        with tf.variable_scope(scope + "/dense1") as nsc:
            net = tf.layers.dropout(net, rate=dropout[8], training=is_training, name="dropout1")
            net = tf.layers.conv2d(net, channel_num[13], kernel_size=[1,1], padding="VALID",
                kernel_regularizer=regularizer(weight_decay))
            self.restore_weights(nsc, "dense", weights_dict)
            net = tf.nn.relu(net, name="relu")
            net = tf.layers.batch_normalization(net, axis=-1, training=is_training)
            self.restore_weights(nsc, "bn", weights_dict)
        with tf.variable_scope(scope + "/dense2") as nsc:
            net = tf.layers.dropout(net, rate=dropout[9], name="dropout2", training=is_training)
            net = tf.layers.conv2d(net, num_classes, kernel_size=[1,1], padding="VALID",
                kernel_regularizer=regularizer(weight_decay))
            self.restore_weights(nsc, "dense", weights_dict)
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.constant(0., dtype=tf.float32))
        if kargs.get("get_features"):
            return [net,] + feature_res
        else:
            return net
