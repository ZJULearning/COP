import collections
import numpy as np
import tensorflow as tf

from .classification_base import ClassificationBase

class ResNet18(ClassificationBase):
    def get_params_and_calculation_from_channel_num(self, channel_num, num_classes, ori_size):
        """
        example: len([16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                          32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                          64, 64, 64, 64, 64, 64, 64, 64, 64, 64])==31
        """
        def get_input_size(index):
            size = ori_size
            if not isinstance(size, int):
                size = size[0]
            if index >= 0 and index <= 10:
                return size
            elif index >= 11 and index <= 20:
                return size / 2
            elif index >= 21 and index <= 30:
                return size / 4
            else:
                raise ValueError("unknown layer index")
            return size

        projection_shortcut_index = [10, 20]

        ## params
        params = 0
        for i, output_channel in enumerate(channel_num):
            input_channel = 3 if i == 0 else channel_num[i-1]
            kernel_size = 3
            params += kernel_size * kernel_size * input_channel * output_channel + output_channel
            # print("params", params)
        params += channel_num[-1] * num_classes + num_classes # for dense layer
        # print("params", params)
        # params for shortcut
        # for shortcut in projection_shortcut_index:
        #     kernel_size = 1
        #     input_channel = channel_num[shortcut]
        #     output_channel = channel_num[shortcut+2]
        #     params += kernel_size * kernel_size * input_channel * output_channel + output_channel

        ## calculation
        calculation = 0
        for i, output_channel in enumerate(channel_num):
            input_channel = 3 if i == 0 else channel_num[i-1]
            input_size = get_input_size(i)
            kernel_size = 3
            calculation += 2 * (input_size ** 2) * input_channel * (output_channel * (kernel_size ** 2))
            # print("calculation", calculation)
        calculation += 2 * channel_num[-1] * num_classes # for dense layer
        # print("calculation", calculation)
        # calculation for shortcut
        # for shortcut in projection_shortcut_index:
        #     kernel_size = 1
        #     input_size = get_input_size(shortcut)
        #     input_channel = channel_num[shortcut]
        #     output_channel = channel_num[shortcut+2]
        #     calculation += 2 * input_size ** 2 * input_channel * (output_channel * kernel_size * kernel_size)

        print("params: ", params, " calculation: ", calculation)
        return params, calculation

    def get_weights_from_model(self, model_path):
        reader = tf.train.NewCheckpointReader(model_path)
        all_variables = reader.get_variable_to_shape_map()
        print(all_variables.keys())
        kernel_weights = collections.OrderedDict()
        ## conv0
        kernel_weights["conv0/conv2d/kernel"] = reader.get_tensor("resnet18/conv0/conv2d/kernel")
        kernel_weights["conv0/conv2d/beta"] = reader.get_tensor("resnet18/conv0/batch_normalization/beta")
        kernel_weights["conv0/conv2d/gamma"] = reader.get_tensor("resnet18/conv0/batch_normalization/gamma")
        kernel_weights["conv0/conv2d/moving_mean"] = reader.get_tensor("resnet18/conv0/batch_normalization/moving_mean")
        kernel_weights["conv0/conv2d/moving_variance"] = reader.get_tensor("resnet18/conv0/batch_normalization/moving_variance")
        ## block1~3
        for i in range(1, 4):
            # sub_block0~4
            for j in range(2):
                # m1~m2
                for k in range(1, 3):
                    kw_prefix = "block%d/sub_block%d/m%d/conv2d"  % (i, j, k)
                    t_prefix = "resnet18/block%d/sub_block%d/m%d" % (i, j, k)
                    kernel_weights[kw_prefix + "/kernel"] = reader.get_tensor(t_prefix + "/conv2d/kernel")
                    kernel_weights[kw_prefix + "/beta"] = reader.get_tensor(t_prefix + "/batch_normalization/beta")
                    kernel_weights[kw_prefix + "/gamma"] = reader.get_tensor(t_prefix + "/batch_normalization/gamma")
                    kernel_weights[kw_prefix + "/moving_mean"] = reader.get_tensor(t_prefix + "/batch_normalization/moving_mean")
                    kernel_weights[kw_prefix + "/moving_variance" ] = reader.get_tensor(t_prefix + "/batch_normalization/moving_variance")
                # shortcut
                kw_prefix = "block%d/sub_block%d/shortcut/conv2d"  % (i, j)
                s_prefix = "resnet18/block%d/sub_block%d/shortcut" % (i, j)
                if s_prefix + "/conv2d/kernel" in all_variables:
                    kernel_weights[kw_prefix + "/kernel"] = reader.get_tensor(s_prefix + "/conv2d/kernel")
                    kernel_weights[kw_prefix + "/beta"] = reader.get_tensor(s_prefix + "/batch_normalization/beta")
                    kernel_weights[kw_prefix + "/gamma"] = reader.get_tensor(s_prefix + "/batch_normalization/gamma")
                    kernel_weights[kw_prefix + "/moving_mean"] = reader.get_tensor(s_prefix + "/batch_normalization/moving_mean")
                    kernel_weights[kw_prefix + "/moving_variance"] = reader.get_tensor(s_prefix + "/batch_normalization/moving_variance")
        ## dense
        kernel_weights["dense/kernel"] = reader.get_tensor("resnet18/dense/dense/kernel")
        kernel_weights["dense/bias"] = reader.get_tensor("resnet18/dense/dense/bias")
        return kernel_weights

    def restore_weights(self, scope, layer_type, weights_dict):
        """
        prefix: scope[9:]
        layer_type: conv, bn, dense
        """
        if layer_type == "conv" or layer_type == "dense":
            prefix = "/conv2d" if layer_type == "conv" else "/dense"
            saved_kernel = weights_dict.get(scope.name[9:] + "/kernel")
            if saved_kernel is not None:
                weight = tf.get_default_graph().get_tensor_by_name(scope.name + prefix + "/kernel:0")
                weight = tf.assign(weight, saved_kernel)
                tf.add_to_collection("init", weight) # important
            # else:
            #     raise ValueError("unknown saved kernel: " + scope.name[9:] + sk_prefix + "/kernel")
            if layer_type == "dense":
                saved_bias = weights_dict.get(scope.name[9:] + "/bias")
                if saved_bias is not None:
                    bias = tf.get_default_graph().get_tensor_by_name(scope.name + prefix + "/bias:0")
                    bias = tf.assign(bias, saved_bias)
                    tf.add_to_collection("init", bias)
                # else:
                #     raise ValueError("unknown saved bias: " + scope.name[9:] + "/bias")
        elif layer_type == "bn":
            saved_beta = weights_dict.get(scope.name[9:] + "/beta")
            saved_gamma = weights_dict.get(scope.name[9:] + "/gamma")
            saved_moving_mean = weights_dict.get(scope.name[9:] + "/moving_mean")
            saved_moving_variance = weights_dict.get(scope.name[9:] + "/moving_variance")
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
            # else:
            #     raise ValueError("unknown saved bn: " + scope.name[9:] + "/batch_normalization/beta")
        else:
            raise ValueError("unknown layer type")
        return

    def basic_block(self, net, channels, strides, use_bias, is_training, initializer, regularizer, weights_dict, weight_decay):
        """
        channels should be a list of shape [2,]
        """
        origin_input = net
        with tf.variable_scope("m1") as nsc:
            net = tf.layers.conv2d(net, channels[0], [3, 3], strides=strides, padding="same", use_bias=use_bias,
                kernel_initializer=initializer(), kernel_regularizer=regularizer(weight_decay))
            self.restore_weights(nsc, "conv", weights_dict)
            net = tf.layers.batch_normalization(net, axis=-1, training=is_training, epsilon=1e-5, momentum=0.997)
            self.restore_weights(nsc, "bn", weights_dict)
            net = tf.nn.relu(net)
        with tf.variable_scope("m2") as nsc:
            net = tf.layers.conv2d(net, channels[1], [3, 3], strides=1, padding="same", use_bias=use_bias,
                kernel_initializer=initializer(), kernel_regularizer=regularizer(weight_decay))
            self.restore_weights(nsc, "conv", weights_dict)
            net = tf.layers.batch_normalization(net, axis=-1, training=is_training, epsilon=1e-5, momentum=0.997,)
            self.restore_weights(nsc, "bn", weights_dict)
        if strides > 1 or origin_input.get_shape().as_list()[3] != channels[1]:
            with tf.variable_scope("shortcut") as nsc:
                # print(strides)
                # print(origin_input.get_shape().as_list()[3])
                # print(channels[1])
                origin_input = tf.layers.conv2d(origin_input, channels[1], [1, 1], strides=strides, padding="same", use_bias=use_bias,
                    kernel_initializer=initializer(), kernel_regularizer=regularizer(weight_decay))
                self.restore_weights(nsc, "conv", weights_dict)
                origin_input = tf.layers.batch_normalization(origin_input, axis=-1, training=is_training, epsilon=1e-5, momentum=0.997,)
                self.restore_weights(nsc, "bn", weights_dict)
        net += origin_input
        net = tf.nn.relu(net)
        return net

    def network(self, inputs, num_classes, scope, is_training, kargs):
        print("Use ResNet18")
        use_bias = kargs.use_bias
        block_sizes = kargs.block_sizes
        strides = kargs.strides
        initializer = kargs.initializer
        regularizer = kargs.regularizer

        weights_dict = kargs.get("weights_dict") or {}
        weight_decay = kargs.weight_decay
        channel_num = kargs.channels_num

        if isinstance(channel_num, dict):
            channel_num = list(channel_num.values())
            print("Use set channels", channel_num)
        else:
            print("Use ori channels", channel_num)

        with tf.variable_scope(scope + "/conv0") as nsc:
            net = tf.layers.conv2d(inputs, channel_num[0], [7, 7], strides=[2, 2], padding="same", use_bias=use_bias,
                kernel_initializer=initializer(), kernel_regularizer=regularizer(weight_decay))
            self.restore_weights(nsc, "conv", weights_dict)
            net = tf.layers.batch_normalization(net, axis=-1, training=is_training, epsilon=1e-5, momentum=0.997,
                beta_regularizer=regularizer(weight_decay), gamma_regularizer=regularizer(weight_decay))
            self.restore_weights(nsc, "bn", weights_dict)
            net = tf.nn.relu(net)
            net = tf.layers.max_pooling2d(net, [3, 3], strides=[2, 2], padding="SAME")

        for i, num_block in enumerate(block_sizes):
            with tf.variable_scope(scope + "/block%d" % (i+1)) as nsc:
                with tf.variable_scope("sub_block0") as nsc:
                    # print(channels_num[2*i*num_block+1: 2*i*num_block+3])
                    net = self.basic_block(net, channel_num[2*i*num_block+1: 2*i*num_block+3], strides[i], use_bias, is_training, initializer, regularizer, weights_dict, weight_decay)
                for j in range(1, num_block):
                    with tf.variable_scope("sub_block%d" % j) as nsc:
                        # print(channels_num[2*(i*num_block+j)+1: 2*(i*num_block+j)+3])
                        net = self.basic_block(net, channel_num[2*(i*num_block+j)+1: 2*(i*num_block+j)+3], 1, use_bias, is_training, initializer, regularizer, weights_dict, weight_decay)

        with tf.variable_scope(scope + "/dense") as nsc:
            assert net.get_shape().as_list()[2] == 7
            net = tf.reduce_mean(net, [1, 2], keepdims=False)
            net = tf.layers.dense(net, num_classes, use_bias=True,
                kernel_regularizer=regularizer(weight_decay), kernel_initializer=initializer())
            self.restore_weights(nsc, "dense", weights_dict)
        return net
