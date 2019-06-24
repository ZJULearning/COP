from abc import abstractmethod
import tensorflow as tf

from .net_base import NetBase

class ClassificationBase(NetBase):
    @abstractmethod
    def get_params_and_calculation_from_channel_num(self, channel_num, num_classes, ori_size):
        """
        The method return the whole model's parameters and float-point operations.
        Args:
            channel_num: a python-list, which contains the number of channels of the whole model in sequence, 
                e.g. config.py:args.ori_channels_num
            num_classes: equal to datasets/dataset:num_classes
            ori_size: the size of input image, e.g. [224, 224]
        Return:
            [parameters, float-point-operations], a python list
        """
        pass

    @abstractmethod
    def get_weights_from_model(self, model_path):
        """
        Get all the weights of the network:
        Args:
            model_path: the path to store the checkpoint
        Return:
            an ordered dictionary of {layer_name: numpy array weights}
        """
        pass

    @abstractmethod
    def restore_weights(self):
        """
        Assign pretrained weights to new model. Add tf.assign op to tf.collection("init").
        You may wish to refer to how VGG16 implements this method
        Args:
            The function is only called by self.network, so you could customize your arguments as need.
        Return:
            None
        """
        pass

    @abstractmethod
    def network(self, inputs, num_classes, scope, is_training, kargs):
        """
        The inference stage of the network
        Args:
            inputs: input images of shape [batch_size, height, width, channels], a python-list
            num_classes: e.g. 10 for cifar10, 100 for cifar100, 1001 for imagenet, an int
            scope: the name scope, a string
            is_training: True if you are training, False if you are testing, a boolean
            kargs: contains all other parameters needed, a config.py:TrainArgs instance
        Return:
            the last layer's output of the network (logits without softmax)
        """
        pass

    def loss(self, scope, logits, labels):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        losses = tf.get_collection('losses', scope)
        total_loss = tf.add_n(losses, name='total_loss')

        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        re_loss = tf.add_n(regularization_loss)

        total_loss = total_loss + re_loss

        return total_loss, re_loss

    def metric_op(self, logits, labels):
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        top_k_op = tf.cast(top_k_op, tf.int64)
        return top_k_op

    def metric(self):
        pass
