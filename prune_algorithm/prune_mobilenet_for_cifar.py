import collections
import logging
import numpy as np
import copy
from .prune_base import PruneBase
from .prune_common import get_parameters, get_computation, get_weights_importance
from .prune_common import get_correlation, get_cosine_sim, get_inner_product, prune_channel

logging.basicConfig(level=logging.ERROR)

class PruneMobileNetForCifar(PruneBase):
    def __init__(self, weights_dict, **prune_args):
        super(PruneMobileNetForCifar, self).__init__(weights_dict, **prune_args)

    # @abstractmethod
    def get_pruned_cared_weights(self, weights_dict):
        return collections.OrderedDict([(key, weight) for key, weight in weights_dict.items() \
            if "kernel" in key and ("conv2d" in key or "dense" in key)])

    # @abstractmethod
    def _get_parameters_related(self, weight_name, cut_channels):
        parameters = 0
        last_dw_name, next_dw_name = self._get_last_and_next_layer_name(weight_name)
        if "conv0" in weight_name:
            assert last_dw_name is None
            tmp_name, next_pw_name = self._get_last_and_next_layer_name(next_dw_name)
            parameters += get_parameters(self.weights_dict, weight_name, cut_channels, None) # this layer
            parameters += get_parameters(self.weights_dict, next_dw_name, cut_channels, weight_name) # next dw
            parameters += get_parameters(self.weights_dict, next_pw_name, cut_channels, weight_name) # next pw

        elif "conv2d" in weight_name and "conv0" not in weight_name:
            last_pw_name, tmp_name = self._get_last_and_next_layer_name(last_dw_name)
            assert tmp_name == weight_name
            tmp_name, next_pw_name = self._get_last_and_next_layer_name(next_dw_name)
            assert tmp_name == weight_name
            parameters += get_parameters(self.weights_dict, weight_name, cut_channels, last_pw_name) # this layer
            parameters += get_parameters(self.weights_dict, next_dw_name, cut_channels, weight_name)
            if next_pw_name is not None:
                parameters += get_parameters(self.weights_dict, next_pw_name, cut_channels, weight_name)
            else:
                assert "dense" in next_dw_name
        elif "dense" in weight_name:
            last_layer_name = last_dw_name
            next_layer_name = next_dw_name
            parameters += get_parameters(self.weights_dict, weight_name, cut_channels, last_layer_name)
            if next_layer_name is not None:
                parameters += get_parameters(self.weights_dict, next_layer_name, cut_channels, weight_name)
        else:
            raise ValueError("Unknown weight_name in MobileNets when getting parameters related: " + weight_name)
        return parameters

    # @abstractmethod
    def _get_computation_related(self, weight_name, cut_channels):
        computation = 0
        last_dw_name, next_dw_name = self._get_last_and_next_layer_name(weight_name)
        if "conv0" in weight_name:
            assert last_dw_name is None
            tmp_name, next_pw_name = self._get_last_and_next_layer_name(next_dw_name)
            output_size = self._get_output_size_from_layer_name(weight_name)
            computation += get_computation(self.weights_dict, weight_name, cut_channels, None, output_size) # this layer
            output_size = self._get_output_size_from_layer_name(next_dw_name)
            computation += get_computation(self.weights_dict, next_dw_name, cut_channels, weight_name, output_size) # next dw
            output_size = self._get_output_size_from_layer_name(next_pw_name)
            computation += get_computation(self.weights_dict, next_pw_name, cut_channels, weight_name, output_size) # next pw

        elif "block" in weight_name:
            last_pw_name, tmp_name = self._get_last_and_next_layer_name(last_dw_name)
            assert tmp_name == weight_name
            tmp_name, next_pw_name = self._get_last_and_next_layer_name(next_dw_name)
            assert tmp_name == weight_name
            output_size = self._get_output_size_from_layer_name(weight_name)
            computation += get_computation(self.weights_dict, weight_name, cut_channels, last_pw_name, output_size) # this layer
            output_size = self._get_output_size_from_layer_name(next_dw_name)
            computation += get_computation(self.weights_dict, next_dw_name, cut_channels, weight_name, output_size)
            if next_pw_name is not None:
                output_size = self._get_output_size_from_layer_name(next_pw_name)
                computation += get_computation(self.weights_dict, next_pw_name, cut_channels, weight_name, output_size)
            else:
                assert "dense" in next_dw_name 
        elif "dense" in weight_name:
            last_layer_name = last_dw_name
            next_layer_name = next_dw_name
            output_size = self._get_output_size_from_layer_name(weight_name)
            computation += get_computation(self.weights_dict, weight_name, cut_channels, last_layer_name, output_size)
            if next_layer_name is not None:
                output_size = self._get_output_size_from_layer_name(next_layer_name)
                computation += get_computation(self.weights_dict, next_layer_name, cut_channels, weight_name, output_size)
        else:
            raise ValueError("Unknown weight_name in MobileNets when getting computation related")
        return computation

    # @abstractmethod
    def _get_params_and_computation_of_whole_model(self, weights_dict):
        all_computation = 0
        all_params = 0
        for weight_name in weights_dict:
            if "kernel" in weight_name:
                # print(weight_name, weights_dict[weight_name].shape)
                output_size = self._get_output_size_from_layer_name(weight_name)
                computation = get_computation(weights_dict, weight_name, {}, "", output_size)
                params = get_parameters(weights_dict, weight_name, {}, "")
                all_computation += computation
                all_params += params
        return all_params, all_computation

    # @abstractmethod
    def _get_output_size_from_layer_name(self, name):
        if "dense" in name:
            return [self.image_size[0] // 32, self.image_size[1] // 32]
        for i in range(12, 14): # block12~13
            if ("block%d" % i) in name:
                return [self.image_size[0] // 8, self.image_size[1] // 8]
        for i in range(6, 12): # block6~11
            if ("block%d" % i) in name:
                return [self.image_size[0] // 4, self.image_size[1] // 4]
        for i in range(4, 6): # block4~5
            if ("block%d" % i) in name:
                return [self.image_size[0] // 2, self.image_size[1] // 2]
        for i in range(1, 4): # block1~3
            if ("block%d" % i) in name:
                return self.image_size[:2]
        if "conv0" in name:
            return self.image_size[:2]
        raise ValueError("unknown layer name for output size: " + name)

    def _get_last_and_next_block_name(self, weight_name):
        last_block_name, next_block_name = self._get_last_and_next_layer_name(weight_name)
        while last_block_name is not None and "dw" in last_block_name:
            last_block_name, _ = self._get_last_and_next_layer_name(last_block_name)

        while next_block_name is not None and "dw" in next_block_name:
            _, next_block_name = self._get_last_and_next_layer_name(next_block_name)
        return last_block_name, next_block_name

    def get_pruned_weights(self, cut_channels):
        for name, cut_channel in cut_channels.items():
            _, next_layer_name = self._get_last_and_next_block_name(name)
            assert next_layer_name is not None
            ## cut output
            weight = self.pruned_weights_dict[name]
            weight = prune_channel(weight, cut_channel, cut_type='output')
            self.pruned_weights_dict[name] = weight

            ## cut bias, and some others
            cut_content = ["kernel", "bias", "beta", "gamma", "moving_mean", "moving_variance"]
            name = name.rstrip(cut_content[0])
            for content in cut_content[1:]:
                if self.pruned_weights_dict.get(name + content) is not None:
                    weight = self.pruned_weights_dict[name + content]
                    weight = prune_channel(weight, cut_channel, cut_type='flatten')
                    self.pruned_weights_dict[name + content] = weight
            ## cut next input
            if "conv2d" in next_layer_name:
                next_dw_name = next_layer_name.replace("conv2d", "dw")
                next_dw_weight = self.pruned_weights_dict[next_dw_name]
                next_weight = self.pruned_weights_dict[next_layer_name]
                # cut dw weight
                next_dw_weight = prune_channel(next_dw_weight, cut_channel, cut_type='input')
                self.pruned_weights_dict[next_dw_name] = next_dw_weight
                # cut dw bias and dw bn
                next_dw_name = next_dw_name.rstrip("kernel")
                for content in cut_content[1:]:
                    if self.pruned_weights_dict.get(next_dw_name + content) is not None:
                        weight = self.pruned_weights_dict[next_dw_name + content]
                        weight = prune_channel(weight, cut_channel, cut_type='flatten')
                        self.pruned_weights_dict[next_dw_name + content] = weight
                # cut next pw weight
                weight = self.pruned_weights_dict[next_layer_name]
                weight = prune_channel(weight, cut_channel, cut_type='input')
                self.pruned_weights_dict[next_layer_name] = weight
            elif "dense" in next_layer_name:
                weight = self.pruned_weights_dict[next_layer_name]
                weight = np.reshape(weight, [1, 1] + list(weight.shape))
                weight = prune_channel(weight, cut_channel, cut_type='input')
                weight = np.squeeze(weight)
                self.pruned_weights_dict[next_layer_name] = weight
            else:
                raise ValueError("unknown layer name for prune weights: " + next_layer_name)
        # for key, weight in self.pruned_weights_dict.items():
        #     print(key, weight.shape)
        return self.pruned_weights_dict