import collections
import os
import numpy as np
import tensorflow as tf

def get_channels_nums(weights_dict, channel_type='output'):
    dim = 3 if channel_type == 'output' else 2
    channels_nums = collections.OrderedDict()

    for key, weight in weights_dict.items():
        if len(weight.shape) == 2:
            weight = np.reshape(weight, [1,1] + list(weight.shape))
        channels_nums[key] = weight.shape[dim]
    return channels_nums

def _get_parameters(kernel_size, input_channel, output_channel):
    params = 0
    params += kernel_size[0] * kernel_size[1] * input_channel * output_channel + output_channel
    return params

def get_parameters(weights_dict, weight_name, cut_channels, last_weight_name):
    this_weight = weights_dict[weight_name]
    if(len(this_weight.shape) == 2):
        this_weight = np.reshape(this_weight, [1, 1] + list(this_weight.shape))
    cut_input_channel = len(cut_channels[last_weight_name]) if last_weight_name in cut_channels else 0 # e.g., last_weight_name is None
    cut_output_channel = len(cut_channels[weight_name]) if weight_name in cut_channels else 0 # e.g., depth-wise layer
    kernel_size = this_weight.shape[:2]
    input_channel = this_weight.shape[2] - cut_input_channel
    output_channel = this_weight.shape[3] - cut_output_channel
    return _get_parameters(kernel_size, input_channel, output_channel)

def _get_computation(output_size, kernel_size, input_channel, output_channel):
    computation = 0
    # FLOPs for input
    computation += 2 * output_size[0] * output_size[1] * output_channel * \
        (input_channel * kernel_size[0] * kernel_size[1])
    # FLOPs for this layer's bn
    # computation += 2 * output_size[0] * output_size[1] * output_channel
    return computation

def get_computation(weights_dict, weight_name, cut_channels, last_weight_name, output_size):
    this_weight = weights_dict[weight_name]
    if(len(this_weight.shape) == 2):
        this_weight = np.reshape(this_weight, [1, 1] + list(this_weight.shape))
    cut_input_channel = len(cut_channels[last_weight_name]) if last_weight_name in cut_channels else 0
    cut_output_channel = len(cut_channels[weight_name]) if weight_name in cut_channels else 0

    kernel_size = this_weight.shape[:2]
    input_channel = this_weight.shape[2] - cut_input_channel
    output_channel = this_weight.shape[3] - cut_output_channel
    
    return _get_computation(output_size, kernel_size, input_channel, output_channel)

def get_normalized_feature(only_weights_dict, impt_method, normalize_method):
    ## compute the statistic feature of filters, e.g., correlation, cosine similarity, ...
    all_stat_dicts = collections.OrderedDict()
    last_name = None
    ## choose the method of computing statistic feature
    if impt_method == "correlation":
        stat_func = get_correlation
    elif impt_method == "cosine":
        stat_func = get_cosine_sim
    elif impt_method == "inner_product":
        stat_func = get_inner_product
    else:
        raise ValueError("Unknown importance metric")
    ## choose the method of normalizing
    if normalize_method == "max":
        norm_func = lambda stat: np.unique(stat)[-1]
    elif normalize_method == "l1":
        norm_func = lambda stat: np.sum(np.abs(stat)) / 2
    elif normalize_method == "l2":
        norm_func = lambda stat: np.linalg.norm(stat)
    else:
        raise ValueError("Unknown normalization method")
    ## compute the feature
    for i, name in enumerate(only_weights_dict.keys()):
        if i == 0:
            last_name = name
        else:
            weight = only_weights_dict[name]
            stat = stat_func(weight)
            all_stat_dicts[last_name] = stat
            last_name = name

    ## normalize feature
    for name, stat in all_stat_dicts.items():
        tmp_stat = stat * (1 - np.eye(stat.shape[0])) # set the numbers in principal diagonal to 0
        normalizer = norm_func(tmp_stat)
        all_stat_dicts[name] = stat / normalizer
    return all_stat_dicts

def get_weights_importance(stat_dicts, top_k, computation_dicts, params_dicts, impt_coefficient):
    assert len(stat_dicts) == len(computation_dicts) and len(stat_dicts) == len(params_dicts)
    ## stat part
    inner_stat_dicts = collections.OrderedDict() # layer_num, channel_num
    for name, stat in stat_dicts.items():
        mean_stat = np.sort(stat, axis=0)
        mean_stat = mean_stat[-top_k-1:-1] # the largest is itself
        mean_stat = np.mean(mean_stat, axis=0, keepdims=False)
        inner_stat_dicts[name] = mean_stat

    ## calculation part
    inner_calculation = np.log2(np.array(list(computation_dicts.values())))
    inner_calculation = inner_calculation / np.max(inner_calculation)
    inner_computation_dicts = collections.OrderedDict(zip(computation_dicts.keys(), inner_calculation))

    ## params part
    inner_params = np.log2(np.array(list(params_dicts.values())))
    inner_params = inner_params / np.max(inner_params)
    inner_params_dicts = collections.OrderedDict(zip(params_dicts.keys(), inner_params))

    ## all importance
    alpha = impt_coefficient[0]
    beta = impt_coefficient[1]
    gamma = impt_coefficient[2]
    importance_dict = collections.OrderedDict()

    for name, stat in inner_stat_dicts.items():
        inner_calculation = inner_computation_dicts[name]
        inner_params = inner_params_dicts[name]
        importance_dict[name] = alpha * (1-stat) + beta * (1-inner_calculation) + gamma * (1-inner_params)
    return importance_dict

def get_correlation(weights):
    """
    weights: of shape [h, w, input_channel, output_channel] or [input_nodes, output_nodes]
    return: of shape [input_channle, input_channel]
    """
    if len(weights.shape) == 2:
        weights = np.reshape(weights, [1, 1] + list(weights.shape)) # dense could be seen as conv whose kernel is [1,1]
    shape = weights.shape
    weights = np.reshape(weights, [shape[0] * shape[1], shape[2], shape[3]]) # combine h and w
    new_shape = weights.shape # of shape h*w, input_channel, output_channel
    feature_mean = np.mean(weights, axis=2, keepdims=True) # h*w, input_channel, 1
    feature_std = np.std(weights, axis=2, keepdims=True) # h*w, input_channel, 1

    feature = weights - feature_mean # of shape h*w, input_channel, output_channel
    feature_t = np.transpose(feature, [0, 2, 1])
    feature_std_t = np.transpose(feature_std, [0, 2, 1])
    # corr: of shape h*w, input_channel, input_channel
    corr = np.matmul(feature, feature_t) / new_shape[2] / (np.matmul(feature_std, feature_std_t) + 1e-8)
    corr = np.abs(corr)
    mean_corr = np.mean(corr, axis=0, keepdims=False) # of shape input_channel, input_channel
    return mean_corr

def get_cosine_sim(weights):
    """
    weights: of shape [h, w, input_channel, output_channel] or [input_nodes, output_nodes]
    return: of shape [input_channle, input_channel]
    """
    if len(weights.shape) == 2:
        weights = np.reshape(weights, [1, 1] + list(weights.shape)) # dense could be seen as conv whose kernel is [1,1]
    shape = weights.shape
    weights = np.reshape(weights, [shape[0] * shape[1], shape[2], shape[3]]) # combine h and w
    feature = weights
    feature_t = np.transpose(feature, [0, 2, 1])

    norm = np.linalg.norm(feature, ord=2, axis=2, keepdims=True)
    norm_t = np.transpose(norm, [0, 2, 1])

    sim = np.matmul(feature, feature_t) / (np.matmul(norm, norm_t) + 1e-8)
    sim = np.abs(sim)
    mean_sim = np.mean(sim, axis=0, keepdims=False)
    return mean_sim

def get_inner_product(weights):
    """
    weights: of shape [h, w, input_channel, output_channel] or [input_nodes, output_nodes]
    return: of shape [input_channle, input_channel]
    """
    if len(weights.shape) == 2:
        weights = np.reshape(weights, [1, 1] + list(weights.shape)) # dense could be seen as conv whose kernel is [1,1]
    shape = weights.shape
    weights = np.reshape(weights, [shape[0] * shape[1], shape[2], shape[3]]) # combine h and w
    feature = weights
    feature_t = np.transpose(feature, [0, 2, 1])
    sim = np.matmul(feature, feature_t)
    sim = np.abs(sim)

    mean_sim = np.mean(sim, axis=0, keepdims=False)
    return mean_sim

def prune_channel(weight, cut_channel, cut_type='input'):
    """
    weight: of shape[h, w, input_channel, output_channel]
    cut_channle: list of numbers to be cut
    """
    if cut_type == 'input':
        weight = np.delete(weight, cut_channel, axis=2) # 4-dimension
    elif cut_type == "output":
        weight = np.delete(weight, cut_channel, axis=3) # 4-dimension
    elif cut_type == "bias" or cut_type == "beta" or cut_type == "gamma" or \
        cut_type == "moving_mean" or cut_type == "moving_variance" or cut_type == "flatten":
        weight = np.delete(weight, cut_channel, axis=None) # 1-dimension
    else:
        raise ValueError("unknown cut type")
    return weight
