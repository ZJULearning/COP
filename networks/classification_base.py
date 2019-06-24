from abc import abstractmethod
import tensorflow as tf

from .net_base import NetBase

class ClassificationBase(NetBase):
    @abstractmethod
    def get_params_and_calculation_from_channel_num(self, channel_num, num_classes, ori_size):
        pass

    @abstractmethod
    def get_weights_from_model(self, model_path):
        pass

    @abstractmethod
    def restore_weights(self, scope, layer_type, weights_dict):
        pass

    @abstractmethod
    def network(self, inputs, num_classes, scope, is_training, kargs):
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
