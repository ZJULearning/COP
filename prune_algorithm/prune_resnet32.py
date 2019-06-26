import collections
import logging
import numpy as np
import copy
import re
from .prune_base import PruneBase
from .prune_common import get_parameters, get_computation, get_weights_importance
from .prune_common import get_correlation, get_cosine_sim, get_inner_product, prune_channel

class PruneResNet32(PruneBase):
    def __init__(self, weights_dict, **prune_args):
        super(PruneResNet32, self).__init__(weights_dict, **prune_args)
        self.merge_all = prune_args["merge_all"]
        self.block_nums = 3

    # @abstractmethod
    def _get_parameters_related(self, weight_name, cut_channels):
        parameters = 0
        last_layer_name, next_layer_name = self._get_last_and_next_block_name(weight_name)
        parameters += get_parameters(self.weights_dict, weight_name, cut_channels, last_layer_name)
        if next_layer_name is not None:
            parameters += get_parameters(self.weights_dict, next_layer_name, cut_channels, weight_name)
        return parameters

    # @abstractmethod
    def _get_computation_related(self, weight_name, cut_channels):
        computation = 0
        last_layer_name, next_layer_name = self._get_last_and_next_block_name(weight_name)
        assert last_layer_name is None or "shortcut" not in last_layer_name
        assert next_layer_name is None or "shortcut" not in next_layer_name
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
        if name.startswith("conv0") or name.startswith("block1"):
            return self.image_size
        elif name.startswith("block2"):
            return [self.image_size[0]//2, self.image_size[1]//2]
        elif name.startswith("block3"):
            return [self.image_size[0]//4, self.image_size[1]//4]
        elif name.startswith("dense"):
            return [self.image_size[0]//32, self.image_size[1]//32]
        else:
            raise ValueError("unknown layer name " + name)

    # @abstractmethod
    def get_pruned_cared_weights(self, weights_dict):
        return collections.OrderedDict([(key, weight) for key, weight in weights_dict.items() \
            if "kernel" in key and "shortcut" not in key])

    def _get_last_and_next_block_name(self, weight_name):
        last_layer_name, next_layer_name = self._get_last_and_next_layer_name(weight_name)
        while last_layer_name is not None and "shortcut" in last_layer_name:
            last_layer_name, _ = self._get_last_and_next_layer_name(last_layer_name)
        while next_layer_name is not None and "shortcut" in next_layer_name:
            _, next_layer_name = self._get_last_and_next_layer_name(next_layer_name)
        return last_layer_name, next_layer_name

    # @abstractmethod
    def get_pruned_weights(self, cut_channels, version):
        # print(self.pruned_weights_dict.keys())
        for name, cut_channel in cut_channels.items():
            _, next_layer_name = self._get_last_and_next_block_name(name)
            next_layer_names = [next_layer_name]
            next_peer_name = next_layer_name.replace("m1", "shortcut") if "sub_block0/m1" in next_layer_name else None
            if next_peer_name in self.pruned_weights_dict:
                next_layer_names.append(next_peer_name)
            # print(name, next_layer_names)
            assert next_layer_name is not None

            ## cut the output channel of this layer
            this_layers_names = [name]
            # peer_name is the linear projection in ResNet, only sub_block0 has liear projection
            peer_name = name.replace("m2", "shortcut") if "sub_block0/m2" in name else None
            if peer_name is not None:
                this_layers_names.append(peer_name)

            cut_content = ["kernel", "bias", "beta", "gamma", "moving_mean", "moving_variance"]
            for name in this_layers_names:
                # cut output
                weight = self.pruned_weights_dict[name]
                weight = prune_channel(weight, cut_channel, cut_type='output')
                self.pruned_weights_dict[name] = weight
                name = name.rstrip(cut_content[0])
                if self.pruned_weights_dict.get(name + cut_content[1]) is not None:
                    weight = self.pruned_weights_dict[name + cut_content[1]]
                    weight = prune_channel(weight, cut_channel, cut_type='flatten')
                    self.pruned_weights_dict[name + cut_content[1]] = weight
                # for resnet_v1, bn layers are just after the cnn layer
                if version == 1:
                    for content in cut_content[2:]:
                        if self.pruned_weights_dict.get(name + content) is not None:
                            weight = self.pruned_weights_dict[name + content]
                            weight = prune_channel(weight, cut_channel, cut_type='flatten')
                            self.pruned_weights_dict[name + content] = weight

            ## cut the input channel of next layer
            for next_name in next_layer_names:
                if "conv2d" in next_name:
                    weight = self.pruned_weights_dict[next_name]
                    weight = prune_channel(weight, cut_channel, cut_type='input')
                    self.pruned_weights_dict[next_name] = weight
                elif "dense" in next_name:
                    weight = self.pruned_weights_dict[next_name]
                    weight = np.reshape(weight, [1, 1] + list(weight.shape))
                    weight = prune_channel(weight, cut_channel, cut_type='input')
                    weight = np.squeeze(weight)
                    self.pruned_weights_dict[next_name] = weight
                else:
                    raise ValueError("unknown next layer name: " + next_name)
                if version == 2 and "shortcut" not in next_name:
                    # batch normalization
                    next_name = next_name.rstrip("kernel")
                    for content in cut_content[2:]: # [beta, gamma, moving_mean, moving_variance]
                        # print(next_name + content)
                        weight = self.pruned_weights_dict[next_name + content]
                        weight = prune_channel(weight, cut_channel, cut_type='flatten')
                        self.pruned_weights_dict[next_name + content] = weight
        return self.pruned_weights_dict

    def __find_subset(self, importances, block_name, merge_all):
        """
        find all sub_block/m1 which contains block_name
        """
        subset = collections.OrderedDict()
        for name in importances.keys():
            add_subset = ("m2" in name) or (merge_all and "m1" in name)
            if name.startswith(block_name) and add_subset and "kernel" in name:
                subset[name] = importances[name]
        return subset

    def __mean_importances(self, importances, subset, is_last_block, merge_all):
        keys = list(subset.keys())
        values = list(subset.values())
        rate = 2 if merge_all else 1
        if is_last_block:
            assert "sub_block%d" % (len(keys) // rate - 1) in keys[-1]
            values = values[:-1]
        mean_values = np.mean(values, axis=0).tolist()
        assert len(mean_values) == len(values[0])
        for name in subset:
            subset[name] = mean_values
            importances[name] = mean_values
        return importances, subset

    def _importance_hook(self, importances):
        for i in range(1, self.block_nums + 1):
            block_name = "block%d" % i
            subset = self.__find_subset(importances, block_name, self.merge_all)
            # print("subset%d" % i, subset.keys())
            importances, _ = self.__mean_importances(importances, subset, i == self.block_nums, self.merge_all)
        return importances

    def _get_cut_layers_name_list(self, cut_channels, cut_layer_name):
        all_cut_layers = []
        if "m2" in cut_layer_name: # or (self.merge_all and "m1" in cut_layer_name)
            index = 0
            while True:
                name = re.sub(r"sub_block\d", "sub_block%d" % index, cut_layer_name)
                m_name = name.replace("m2", "m1")
                if self.merge_all and m_name in cut_channels:
                    all_cut_layers.append(m_name)
                if name in cut_channels:
                    all_cut_layers.append(name)
                else:
                    break
                index += 1
        elif self.merge_all and "m1" in cut_layer_name:
            index = 0
            while True:
                name = re.sub(r"sub_block\d", "sub_block%d" % index, cut_layer_name)
                m_name = name.replace("m1", "m2")
                if name in cut_channels and m_name in cut_channels:
                    all_cut_layers.append(name)
                    all_cut_layers.append(m_name)
                else:
                    break
                index += 1
        else:
            all_cut_layers = [cut_layer_name]
        # print(all_cut_layers)
        return all_cut_layers
