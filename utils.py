import torch
import os
import logging 
from tabulate import tabulate

OP_NAMES = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

EDGE_OPS = {
    'TF-MOTNAS-A': 
    {'full': ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"],
    'none_conv1x1_conv3x3': ["none", "nor_conv_1x1", "nor_conv_3x3"],
    'skip_avg': ["skip_connect", "avg_pool_3x3"],
    'none_conv1x1': ["none", "nor_conv_1x1"],
    'none_conv3x3': ["none", "nor_conv_3x3"],
    'none': ["none"],
    'conv1x1': ["nor_conv_1x1"],
    'conv3x3': ["nor_conv_3x3"],
    'skip': ["skip_connect"],
    'pool': ["avg_pool_3x3"]},

    'TF-MOTNAS-B': 
    {'full': ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"],
    'none': ["none"],
    'not_none': ["skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"],
    'conv': ["nor_conv_1x1", "nor_conv_3x3"],
    'topo': ["skip_connect", "avg_pool_3x3"],
    'conv1x1': ["nor_conv_1x1"],
    'conv3x3': ["nor_conv_3x3"],
    'skip': ["skip_connect"],
    'pool': ["avg_pool_3x3"]}
}

NODE_CHILDREN = {
    'TF-MOTNAS-A': 
        {'full': ["none_conv1x1_conv3x3", "skip_avg"],
       'none_conv1x1_conv3x3': ["none_conv1x1", "none_conv3x3"],
        'none_conv3x3': ["none", "conv3x3"],
        'none_conv1x1': ["none", "conv1x1"],
        'skip_avg': ["skip", "pool"],
        'none': [],
        'conv1x1': [],
        'conv3x3': [],
        'skip': [],
        'pool': []
    },

    'TF-MOTNAS-B': {
        'full': ["none", "not_none"],
        'none': [],
        'not_none': ["conv", "topo"],
        'conv': ["conv1x1", "conv3x3"],
        'topo': ["skip", "pool"],
        'conv1x1': [],
        'conv3x3': [],
        'skip': [],
        'pool': []}
}



def to_string(ind):
    cell = ''
    node = 0
    for i in range(len(ind)):
        gene = ind[i]
        cell += '|' + OP_NAMES[gene] + '~' + str(node)
        node += 1
        if i == 0 or i == 2:
            node = 0
            cell += '|+'
    cell += '|'
    return cell

def get_num_classes(dataset):
    return 100 if dataset == 'cifar100' else 10 if dataset == 'cifar10' else 120

def get_input(args, train_loader):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':    
        return torch.randn(len(train_loader), 3, 32, 32).to(args.device)
    else:
        return torch.randn(len(train_loader), 3, 16, 16).to(args.device)

def create_exp_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

def print_table_indicators(args, indicators):
    for indicator_name in indicators:
        logging.info(indicator_name)
        indicator_list = []
        for dataset in args.datasets:
            indicator_temp = list(indicators[indicator_name][dataset].values())
            indicator_temp.insert(0, dataset)
            indicator_list.append(indicator_temp)
        
        headers = ['test accuracy - flops']
        logging.info(tabulate(indicator_list, headers=headers, tablefmt="grid"))

def subnet_to_str(args, subnet):
    position_in_subnet_arr = 0
    arch_str = ""
    for node_in_index in range(1, 4):
        for node_out_index in range(0, node_in_index):
            for edge_name in EDGE_OPS[args.method][subnet[position_in_subnet_arr]]:
                arch_str = arch_str + "|" + edge_name + "~" + str(node_out_index)
            position_in_subnet_arr += 1
        arch_str = arch_str + "|+"
    return arch_str[0:-1]

