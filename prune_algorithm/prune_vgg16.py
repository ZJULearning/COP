import collections
import logging
import numpy as np
import copy
from .prune_base import PruneBase
from .prune_common import get_parameters, get_computation, get_weights_importance
from .prune_common import get_correlation, get_cosine_sim, get_inner_product, prune_channel

logging.basicConfig(level=logging.ERROR)

class PruneVgg16(PruneBase):
    def __init__(self, weights_dict, **prune_args):
        super(PruneVgg16, self).__init__(weights_dict, **prune_args)

    # @abstractmethod
    def get_pruned_cared_weights(self, weights_dict):
        return collections.OrderedDict([(key, weight) for key, weight in weights_dict.items() if "kernel" in key])

    # @abstractmethod
    def _get_parameters_related(self, weight_name, cut_channels):
        parameters = 0
        last_layer_name, next_layer_name = self._get_last_and_next_layer_name(weight_name)
        parameters += get_parameters(self.weights_dict, weight_name, cut_channels, last_layer_name)
        if next_layer_name is not None:
            parameters += get_parameters(self.weights_dict, next_layer_name, cut_channels, weight_name)
        return parameters

    # @abstractmethod
    def _get_computation_related(self, weight_name, cut_channels):
        computation = 0
        last_layer_name, next_layer_name = self._get_last_and_next_layer_name(weight_name)
        output_size = self._get_output_size_from_layer_name(weight_name)
        computation += get_computation(self.weights_dict, weight_name, cut_channels, last_layer_name, output_size)
        if next_layer_name is not None:
            output_size = self._get_output_size_from_layer_name(next_layer_name)
            computation += get_computation(self.weights_dict, next_layer_name, cut_channels, weight_name, output_size)
        return computation

    # @abstractmethod
    def _get_params_and_computation_of_whole_model(self, weights_dict):
        all_computation = 0
        all_params = 0
        for weight_name in weights_dict:
            if ("conv" in weight_name or "dense" in weight_name) and "kernel" in weight_name:
                output_size = self._get_output_size_from_layer_name(weight_name)
                computation = get_computation(weights_dict, weight_name, {}, "", output_size)
                params = get_parameters(weights_dict, weight_name, {}, "")
                all_computation += computation
                all_params += params
        return all_params, all_computation

    # @abstractmethod
    def _get_output_size_from_layer_name(self, name):
        if "conv1" in name:
            return self.image_size
        elif "conv2" in name:
            return [self.image_size[0]//2, self.image_size[1]//2]
        elif "conv3" in name:
            return [self.image_size[0]//4, self.image_size[1]//4]
        elif "conv4" in name:
            return [self.image_size[0]//8, self.image_size[1]//8]
        elif "conv5" in name:
            return [self.image_size[0]//16, self.image_size[1]//16]
        elif "dense" in name:
            return [self.image_size[0]//32, self.image_size[1]//32]
        else:
            raise ValueError("unknown layer name " + name)

    def get_pruned_weights(self, cut_channels):
        for name, cut_channel in cut_channels.items():
            _, next_layer_name = self._get_last_and_next_layer_name(name)
            cut_content = ["kernel", "bias", "beta", "gamma", "moving_mean", "moving_variance"]
            # if next_layer_name is not None:
            assert next_layer_name is not None
            # cut output
            weight = self.pruned_weights_dict[name]
            weight = prune_channel(weight, cut_channel, cut_type='output')
            self.pruned_weights_dict[name] = weight
            name = name.rstrip(cut_content[0])
            # cut bias and bn parameters
            for content in cut_content[1:]:
                if self.pruned_weights_dict.get(name + content) is not None:
                    weight = self.pruned_weights_dict[name + content]
                    weight = prune_channel(weight, cut_channel, cut_type='flatten')
                    self.pruned_weights_dict[name + content] = weight
            # cut next input
            weight = self.pruned_weights_dict[next_layer_name]
            weight = prune_channel(weight, cut_channel, cut_type='input')
            self.pruned_weights_dict[next_layer_name] = weight
        return self.pruned_weights_dict
