import os
import re
import logging
import collections
import functools
import numpy as np

def _find_subset(importances, block_name, merge_all):
    """
    find all sub_block/m1 which contains block_name
    """
    subset = collections.OrderedDict()
    for name in importances.keys():
        add_subset = ("m2" in name) or (merge_all and "m1" in name)
        if name.startswith(block_name) and add_subset and "conv2d/kernel" in name:
            subset[name] = importances[name]
    return subset

def _mean_importances(importances, subset, is_last_block, merge_all):
    """
    mean subset importances
    ignore fully connected layer for last block
    """
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

def _merge_channels(cut_channels, subset):
    values = list(subset.values())
    intersect = list(functools.reduce(np.intersect1d, values, values[0]))
    # assert len(intersect) == len(values[0])
    for name in subset:
        subset[name] = intersect
        cut_channels[name] = intersect
    return cut_channels, subset

def merge_importances(importances, block_nums, merge_all):
    """
    importances is a orderedDict
    """
    for i in range(1, block_nums + 1):
        block_name = "block%d" % i
        subset = _find_subset(importances, block_name, merge_all)
        # print("subset%d" % i, subset.keys())
        importances, _ = _mean_importances(importances, subset, i == block_nums, merge_all)
    return importances

def merge_cut_layers(cut_channels, cut_layer_name, is_merge_importance, merge_all):
    all_cut_layers = []
    if ("m2" in cut_layer_name) or (merge_all and "m1" in cut_layer_name) and is_merge_importance:
        index = 0
        while True:
            name = re.sub(r"sub_block\d", "sub_block%d" % index, cut_layer_name)
            name = name.replace("m1", "m2")
            m_name = name.replace("m2", "m1")
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

def merge_cut_channels(cut_channels, is_merge_importance, block_nums, merge_all):
    if is_merge_importance:
        return cut_channels
    for i in range(1, block_nums + 1):
        block_name = "block%d" % i
        subset = _find_subset(cut_channels, block_name, merge_all)
        cut_channels, _ = _merge_channels(cut_channels, subset)
    return cut_channels


