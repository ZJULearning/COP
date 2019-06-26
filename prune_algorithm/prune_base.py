from abc import ABC, abstractmethod
import copy
import collections
import numpy as np

from . import prune_common
from .prune_common import get_parameters, get_computation, get_weights_importance
from .prune_common import get_correlation, get_cosine_sim, get_inner_product

class PruneBase(ABC):
    def __init__(self, weights_dict, **prune_args):
        self.image_size = prune_args["image_size"]
        self.importance_method = prune_args["importance_method"]
        self.importance_coefficient = prune_args["importance_coefficient"]
        self.top_k = prune_args["top_k"]
        self.num_classes = prune_args["num_classes"]
        self.normalize_method = prune_args["normalize_method"]
        self.conv_dense_separate = prune_args["conv_dense_separate"]

        self.weights_dict = weights_dict
        self.pruned_weights_dict = copy.deepcopy(self.weights_dict)

    @abstractmethod
    def _get_parameters_related(self, weight_name, cut_channels):
        """
        Get the parameters related to "weight_name". For example:
            layer1 and layer2 are two consecutive layers. The number of output channels of layer1
            influences the parameters of layer1 and layer2 simultaneously. So if 
            weight_name="layer1", the method would return the sum of parameters of layer1 and layer2
        Args:
            weight_name: the layer name, e.g., "conv1"
            cut_channels: an ordered dictionary of {layer name: [the indices of pruned channels of the layer]}
        Return:
            an integer, the number of parameters
        """
        pass

    @abstractmethod
    def _get_computation_related(self, weight_name, cut_channels):
        """
        The same as _get_parameters_related, but this method return the number of float-point operations
        Return:
            an integer, the number of float-point operations
        """
        pass

    @abstractmethod
    def _get_params_and_computation_of_whole_model(self, weights_dict):
        """
        Get the number of parameters and float-points operations of the whole model
        Args:
            weights_deict: an ordered dictionary, all the weights of the model
        Return:
            a list of 2 integers: [parameters, float-point operations]
        """
        pass

    @abstractmethod
    def _get_output_size_from_layer_name(self, name):
        """
        Get the size of output feature maps of a certain layer
        Args:
            name: layer name
        Return:
            a list of 2 integers: [height, width]
        """
        pass

    @abstractmethod
    def get_pruned_cared_weights(self, weights_dict):
        """
        Get the weights of the layers you want to prune
        Args:
            weights_dict: an ordered dictionary, all the weights of the model
        Return:
            an ordered dictionary, the weights we care during pruning. For example, it will not contain weights
                of depth-wise layer for mobilenets
        """
        pass

    @abstractmethod
    def get_pruned_weights(self):
        """
        Get numpy-type weights tensor after pruned
        Return:
            an ordered dictionary: {layer name: a numpy tensor of weights}
        """
        pass

    def _importance_hook(self, importances):
        """
        some extra code after computing the importance of filters. For example, we need to compute
         the average importance of some layers in ResNet, see prune_resnet*.py or our paper for details
        """
        return importances

    def _get_cut_layers_name_list(self, cut_channels, cut_layer_name):
        return [cut_layer_name]

    def get_channels_nums(self, weights_dict, channel_type):
        """
        Get the number of channels of each layer.
        Args:
            weights_dict: an ordered dictionary, all the weights of the model
            channel_type: 'output' or 'input', whether get the number of input or output channels
        Return:
            an ordered dictionary, whose keys are the same as weights_dict, {layer name: the number of channels}
        """
        return prune_common.get_channels_nums(weights_dict, channel_type)

    def _get_normalized_feature(self, cared_weights_dict):
        """
        Get the importance of all the cared weights
        Args:
            cared_weights_dict: an ordered dictionary, the weights of the model
        """
        return prune_common.get_normalized_feature(cared_weights_dict, self.importance_method, self.normalize_method)

    def _get_last_and_next_layer_name(self, weight_name):
        """
        Get the name of a certain name's last and next layer
        Args:
            weight_name: layer name
        Return:
            a list of 2 strings [its_last_layer_name, its_next_layer_name]
        """
        keys = list(self.weights_dict.keys())
        ## find last layerf
        this_weight_index = keys.index(weight_name)
        for i in range(this_weight_index-1, -1, -1):
            if "kernel" in keys[i]:
                last_layer_name = keys[i]
                break
        else:
            last_layer_name = None
        ## find next layer
        for i in range(this_weight_index+1, len(keys)):
            if "kernel" in keys[i]:
                next_layer_name = keys[i]
                break
        else:
            next_layer_name = None
        return last_layer_name, next_layer_name

    def _get_last_and_next_block_name(self, weight_name):
        """
        For MobileNets, it returns the last and next point-wise layer names
        For others, it returns the last_and_next_layer_name
        """
        return self._get_last_and_next_layer_name(weight_name)

    def _iteratively_prune(self, cared_weights_dict, stat_dicts, cut_nums):
        cut_channels = collections.OrderedDict()
        for key in stat_dicts.keys():
            cut_channels[key] = []

        ## compute calculation and params for all layers
        computation_dicts = collections.OrderedDict()
        params_dicts = collections.OrderedDict()
        for weight_name in list(cared_weights_dict.keys())[:-1]:
            params_dicts[weight_name] = self._get_parameters_related(weight_name, cut_channels)
            computation_dicts[weight_name] = self._get_computation_related(weight_name, cut_channels)

        # print(stat_dicts.keys())
        # print(computation_dicts.keys())
        # print(params_dicts.keys())
        ## prune iteratively
        i = 0
        while i < cut_nums:
            ## find min imporantce
            importances = get_weights_importance(stat_dicts, self.top_k, computation_dicts, params_dicts, self.importance_coefficient)
            importances = self._importance_hook(importances) # for network with residual design, e.g., ResNet
            for key, value in cut_channels.items():
                if importances.get(key) is not None:
                    # print(key, value)
                    for channel in value:
                        importances[key][channel] = np.inf
            argmin_within_layers = list(map(np.argmin, list(importances.values())))
            min_within_layers = list(map(np.min, list(importances.values())))
            argmin_cross_layers = np.argmin(np.array(min_within_layers))
            ## cut the least important channel
            cut_layer_name = list(importances.keys())[argmin_cross_layers]
            cut_channel_index = argmin_within_layers[argmin_cross_layers]
            # if cut_layer_name == "block1/sub_block0/m1/conv2d/kernel" and cut_channel_index == 9:
            #     print(min_within_layers)
            cut_layers_names = self._get_cut_layers_name_list(cut_channels, cut_layer_name)
            # print(cut_layer_name, cut_layers_names)
            for cut_layer_name in cut_layers_names:
                cut_channels[cut_layer_name].append(cut_channel_index)

                ## set the similarity of the cut channel with others to 0
                stat_dicts[cut_layer_name][cut_channel_index,:] = 0
                stat_dicts[cut_layer_name][:,cut_channel_index] = 0
                ## re-compute the computation and parameters for the layer which is cut
                params_dicts[cut_layer_name] = self._get_parameters_related(cut_layer_name, cut_channels)
                computation_dicts[cut_layer_name] = self._get_computation_related(cut_layer_name, cut_channels)
                last_layer_name, next_layer_name = self._get_last_and_next_block_name(cut_layer_name)
                if last_layer_name in params_dicts: # false when last_layer_name is None or last_layer_name == cared_weights_dict.keys())[-1]
                    params_dicts[last_layer_name] = self._get_parameters_related(last_layer_name, cut_channels)
                    computation_dicts[last_layer_name] = self._get_computation_related(last_layer_name, cut_channels)
                if next_layer_name in params_dicts: # false when next_layer_name is None or last_layer_name == cared_weights_dict.keys())[-1]
                    params_dicts[next_layer_name] = self._get_parameters_related(next_layer_name, cut_channels)
                    computation_dicts[next_layer_name] = self._get_computation_related(next_layer_name, cut_channels)
            i += len(cut_layers_names)
            if (i + 1) % 100 == 0:
                print("cut %d channels finished" % (i+1))
        print("cut %d channels finished" % i)
        return cut_channels

    def get_prune_channels(self, prune_rate):
        """
        Get the indices of channels to be pruned
        Args:
            prune_rate: the prune rate of the whole model
        Return:
            an ordered dictionary: {layer name: the indices of pruned channels of the layer}
        """
        cared_weights_dict = self.get_pruned_cared_weights(self.weights_dict)

        stat_dicts = self._get_normalized_feature(cared_weights_dict)
        # print(self.weights_dict.keys())
        ## compute the number of channels to be pruned
        all_channels_nums = list(self.get_channels_nums(cared_weights_dict, channel_type="input").values())
        conv_channels_nums = [cared_weights_dict[key].shape[2] for key in cared_weights_dict if "conv" in key]
        dense_channels_nums = []
        for key in cared_weights_dict:
            # implement dense layer with 1x1 convolutional layer
            if "dense" in key and len(cared_weights_dict[key].shape) == 4:
                dense_channels_nums.append(cared_weights_dict[key].shape[2])
            # implement dense layer with dense layer
            elif "dense" in key and len(cared_weights_dict[key].shape) == 2:
                dense_channels_nums.append(cared_weights_dict[key].shape[0])
        assert sum(all_channels_nums) == sum(conv_channels_nums) + sum(dense_channels_nums)
        conv_cut_nums = int(sum(conv_channels_nums) * prune_rate)
        dense_cut_nums = int(sum(dense_channels_nums) * prune_rate)
        print("The number of channels to be cut: %d (%d + %d)" % (conv_cut_nums + dense_cut_nums, conv_cut_nums, dense_cut_nums))

        if self.conv_dense_separate:
            pass
        else:
            all_cut_nums = conv_cut_nums + dense_cut_nums
            cut_channels = self._iteratively_prune(cared_weights_dict, stat_dicts, all_cut_nums)

        for name, cut_channel in cut_channels.items():
            cut_channel = sorted(cut_channel)
            print(name, len(cut_channel), cut_channel)
            assert len(set(cut_channel)) == len(cut_channel)

        return cut_channels

    def get_pruned_ratio(self):
        """
        Get the pruned ratio according to the original weights and the pruned weights
        Return:
            Frr: the ratio of float-point operations to be reduced
            Prr: the ratio of parameters to be pruned
        """
        origin_params, origin_computation = self._get_params_and_computation_of_whole_model(self.weights_dict)
        retained_params, retained_computation = self._get_params_and_computation_of_whole_model(self.pruned_weights_dict)
        computation_pruned_ratio = 1 - retained_computation / origin_computation
        pruned_pruned_ratio = 1 - retained_params / origin_params
        print("origin computation: %d, new computation: %d, pruned ratio: %f" % \
            (origin_computation, retained_computation,  computation_pruned_ratio))
        print("origin params : %d, new params: %d, pruned ratio: %f" % \
            (origin_params, retained_params, pruned_pruned_ratio))
        return computation_pruned_ratio, pruned_pruned_ratio