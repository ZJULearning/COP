import collections
import numpy as np
import tensorflow as tf

from .classification_base import ClassificationBase

class MobileNetForCifar(ClassificationBase):
    def get_params_and_calculation_from_channel_num(self, channel_num, num_classes, ori_size):
        """
        example: [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        """
        def get_output_size(index):
            size = ori_size
            if not isinstance(size, int):
                size = size[0]
            if index >= 0 and index <= 3:
                return size
            elif index >= 4 and index <= 5:
                return size / 2
            elif index >= 6 and index <= 11:
                return size / 4
            elif index >= 12 and index <= 13:
                return size / 8
            return size # will never be run

        ## for params
        params = 0
        params += 3 * 3 * 3 * channel_num[0] + channel_num[0] # for conv0
        # print("conv0 params: ", params)
        for i in range(1, len(channel_num)):
            params += 3 * 3 * channel_num[i-1] * 1 + 1 # for dw
            params += 1 * 1 * channel_num[i-1] * channel_num[i] + channel_num[i] # for pw
            # print("block%d params: " % i, params)
        params += channel_num[-1] * num_classes + num_classes
        # print("dense params: ", params)

        ## for calculation
        calculation = 0
        input_size = get_output_size(0)
        calculation += 2 * input_size ** 2 * channel_num[0] * (3 * 3 * 3)
        # print("conv0 cal: ", calculation)
        for i in range(1, len(channel_num)):
            output_size = get_output_size(i)
            calculation += 2 * output_size ** 2 * channel_num[i-1] * (1 * 3 * 3) # for dw
            calculation += 2 * output_size ** 2 * channel_num[i-1] * (channel_num[i] * 1 * 1) # for pw
            # print("block%d cal: " % i, calculation)
        calculation += 2 * num_classes * channel_num[-1]
        # print("dense cal: ", calculation)
        print("params: ", params, " calculation: ", calculation)
        return params, calculation

    def get_weights_from_model(self, model_path):
        """
        return weights_dict
        """
        reader = tf.train.NewCheckpointReader(model_path)
        all_variables = reader.get_variable_to_shape_map()
        kernel_weights = collections.OrderedDict()
        ## conv0
        kernel_weights["conv0/conv2d/kernel"] = reader.get_tensor("conv0/conv2d/kernel")
        kernel_weights["conv0/conv2d/beta"] = reader.get_tensor("conv0/batch_normalization/beta")
        kernel_weights["conv0/conv2d/gamma"] = reader.get_tensor("conv0/batch_normalization/gamma")
        kernel_weights["conv0/conv2d/moving_mean"] = reader.get_tensor("conv0/batch_normalization/moving_mean")
        kernel_weights["conv0/conv2d/moving_variance"] = reader.get_tensor("conv0/batch_normalization/moving_variance")
        for i in range(13):
            prefix = "block" + str(i+1)
            ## dw conv2d
            kernel_weights[prefix+"/dw/kernel"] = reader.get_tensor(prefix+"/dw/kernel")
            ## dw batch norm
            kernel_weights[prefix+"/dw/beta"] = reader.get_tensor(prefix+"/dw/batch_normalization/beta")
            kernel_weights[prefix+"/dw/gamma"] = reader.get_tensor(prefix+"/dw/batch_normalization/gamma")
            kernel_weights[prefix+"/dw/moving_mean"] = reader.get_tensor(prefix+"/dw/batch_normalization/moving_mean")
            kernel_weights[prefix+"/dw/moving_variance"] = reader.get_tensor(prefix+"/dw/batch_normalization/moving_variance")
            ## pw conv2d
            kernel_weights[prefix+"/conv2d/kernel"] = reader.get_tensor(prefix+"/conv2d/kernel")
            ## pw batch norm
            kernel_weights[prefix+"/conv2d/beta"] = reader.get_tensor(prefix+"/pw/batch_normalization/beta")
            kernel_weights[prefix+"/conv2d/gamma"] = reader.get_tensor(prefix+"/pw/batch_normalization/gamma")
            kernel_weights[prefix+"/conv2d/moving_mean"] = reader.get_tensor(prefix+"/pw/batch_normalization/moving_mean")
            kernel_weights[prefix+"/conv2d/moving_variance"] = reader.get_tensor(prefix+"/pw/batch_normalization/moving_variance")
        kernel_weights["dense/dense/kernel"] = reader.get_tensor("dense/dense/kernel")
        kernel_weights["dense/dense/bias"] = reader.get_tensor("dense/dense/bias")
        return kernel_weights

    def restore_weights(self, scope, layer_type, weights_dict, infix=None):
        """
        prefix: scope
        layer_type: conv, bn, local,
        -- name in weight_dict: "dw", "conv2d", "dw/beta gamme moving...", "pw/beta gamma moving"
        """
        if layer_type == "conv" or layer_type == "dw" or layer_type == "dense":
            saved_kernel = weights_dict.get(scope.name + infix + "/kernel")
            saved_bias = weights_dict.get(scope.name + infix + "/bias")
            if saved_kernel is not None:
                weight = tf.get_default_graph().get_tensor_by_name(scope.name + infix + "/kernel:0")
                weight = tf.assign(weight, saved_kernel)
                tf.add_to_collection("init", weight)
            if saved_bias is not None:
                bias = tf.get_default_graph().get_tensor_by_name(scope.name + infix + "/bias:0")
                bias = tf.assign(bias, saved_bias)
                tf.add_to_collection("init", bias)
        elif layer_type == "bn":
            saved_beta = weights_dict.get(scope.name + infix + "/beta")
            saved_gamma = weights_dict.get(scope.name + infix + "/gamma")
            saved_moving_mean = weights_dict.get(scope.name + infix + "/moving_mean")
            saved_moving_variance = weights_dict.get(scope.name + infix + "/moving_variance")
            if saved_beta is not None:
                beta = tf.get_default_graph().get_tensor_by_name(scope.name + infix + "/beta:0")
                gamma = tf.get_default_graph().get_tensor_by_name(scope.name + infix + "/gamma:0")
                moving_mean = tf.get_default_graph().get_tensor_by_name(scope.name + infix + "/moving_mean:0")
                moving_variance = tf.get_default_graph().get_tensor_by_name(scope.name + infix + "/moving_variance:0")
                print(beta.get_shape().as_list())
                print(saved_beta.shape)
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

    def block(self, net, name, channels, strides, is_training, initializer, regularizer, weight_decay, use_bias, weights_dict, dropout=None):
        with tf.variable_scope(name) as nsc:
            # depthwise convolution
            kernel_shape = [3, 3, net.get_shape().as_list()[3], 1]
            depth_kernel = tf.get_variable("dw/kernel", kernel_shape, initializer=initializer(), regularizer=regularizer(weight_decay))
            net = tf.nn.depthwise_conv2d(net, depth_kernel, [1,strides,strides,1], padding="SAME")
            self.restore_weights(nsc, "dw", weights_dict, "/dw")

            net = tf.layers.batch_normalization(net, axis=-1, training=is_training, epsilon=1e-5, momentum=0.9, name="dw/batch_normalization")
            self.restore_weights(nsc, "bn", weights_dict, "/dw/batch_normalization")
            net = tf.nn.relu6(net)

            net = tf.layers.conv2d(net, channels, [1, 1], strides=[1, 1], use_bias=use_bias,
                kernel_initializer=initializer(), kernel_regularizer=regularizer(weight_decay))
            self.restore_weights(nsc, "conv", weights_dict, "/conv2d")
            net = tf.layers.batch_normalization(net, axis=-1, training=is_training, epsilon=1e-5, momentum=0.9, name="pw/batch_normalization")
            self.restore_weights(nsc, "bn", weights_dict, "/pw/batch_normalization")
            net = tf.nn.relu6(net)
            # net = tf.layers.dropout(net, rate=dropout, training=is_training)
        return net

    def network(self, inputs, num_classes, scope, is_training, kargs):
        print("Use MobileNet")
        initializer = tf.contrib.layers.variance_scaling_initializer
        regularizer = kargs.regularizer
        weight_decay = kargs.weight_decay
        use_bias = kargs.use_bias

        channel_num = kargs.channels_num
        weights_dict = kargs.get("weights_dict") or {}

        if isinstance(channel_num, dict):
            channel_num = list(channel_num.values())
            print("Use set channels: ", channel_num)
        else:
            print("Use ori channels: ", channel_num)

        strides = [1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
        with tf.variable_scope("conv0") as nsc:
            # conv0
            net = tf.layers.conv2d(inputs, channel_num[0], [3, 3], strides=strides[0], padding="same", use_bias=use_bias,
                kernel_initializer=initializer(), kernel_regularizer=regularizer(weight_decay))
            self.restore_weights(nsc, "conv", weights_dict, "/conv2d")
            net = tf.layers.batch_normalization(net, axis=-1, training=is_training, epsilon=1e-5, momentum=0.9)
            self.restore_weights(nsc, "bn", weights_dict, "/batch_normalization")
            net = tf.nn.relu6(net)
        # block1~n
        for i in range(1, 14):
            channels = channel_num[i]
            net = self.block(net, "block" + str(i), channels, strides[i], is_training=is_training, # dropout=dropout[i],
                initializer=initializer, regularizer=regularizer, weight_decay=weight_decay, use_bias=use_bias, weights_dict=weights_dict)
        with tf.variable_scope("dense") as nsc:
            # avg_pooling
            net = tf.layers.average_pooling2d(net, [4, 4], strides=1, name=scope+"/average_pooling2d")
            # linear
            net = tf.squeeze(net, [1, 2], name=scope+'/squeeze')
            net = tf.layers.dense(net, num_classes, activation=None, use_bias=True,
                kernel_initializer=initializer(), kernel_regularizer=regularizer(weight_decay))
            self.restore_weights(nsc, "dense", weights_dict, "/dense")
        return net
