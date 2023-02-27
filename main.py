import pickle
import json
import argparse
import numpy as np
import torch
import logging
import time
import os
import sys

from utils import *
from indicators import *
from nats_bench import create
from search import tree_search, convert_archive

from foresight.dataset import get_cifar_dataloaders

def parse_arguments():
    parser = argparse.ArgumentParser("Training-free Multi-Objective Tree Neural Architecture Search")
    parser.add_argument('--n_runs', type=int, default=30, help='number of runs')
    parser.add_argument('--continue_run', type=int, default=0, help='continue run')
    parser.add_argument('--performance_obj', type=str, default='zero-cost-proxies', help='performance objective [zero-cost-proxies, synflow, valid-accuracy]')
    parser.add_argument('--method', type=str, default='TF-MOTNAS-A', help='methods [TF-MOTNAS-A, TF-MOTNAS-B]')

    parser.add_argument('--init_w_type', type=str, default='kaiming', help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='zero', help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--init_channels', default=4, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--img_size', default=8, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, imagenet]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')

    args = parser.parse_args()
    args.save = 'experiment/{}/{}'.format(args.method , args.dataset)
    args.objective_list = ['flops', args.performance_obj] if args.performance_obj in ['synflow', 'valid-accuracy'] else ['flops', 'synflow', 'jacob_cov']
    if args.dataset=="ImageNet16-120":
      
        if args.performance_obj=='valid-accuracy':
            args.train_loader, args.val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers, resize=args.img_size, datadir='benchmark/')
        else:
            args.train_loader, args.test_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers, resize=None, datadir='benchmark/')
    else:
        if args.performance_obj=='valid-accuracy':
            args.train_loader, args.val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers, resize=None)
        else:
            args.train_loader, args.test_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers)
    args.api = create("benchmark/NATS-tss-v1_0-3ffb9-simple", 'tss', fast_mode=True, verbose=False)
    args.datasets = ['cifar10', 'cifar100', "ImageNet16-120"] if args.dataset=='cifar10' else [args.dataset]

    args.pf = pickle.load(open('benchmark/pareto_front/pf.pickle', "rb"))
    args.pf_norm = pickle.load(open('benchmark/pareto_front/pf_norm.pickle', "rb"))
    args.max_min_measures = pickle.load(open('benchmark/max_min_measures.pickle', "rb"))
    args.ref_point = [1.05] * 2
    create_exp_dir(args.save)
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    
    return args

if __name__ == '__main__':
    args = parse_arguments()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    n_var = 6 #NATS-Bench

    
    for seed in range(args.continue_run, args.n_runs):
        torch.manual_seed(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        log_run = {
            'time': [],
            'archive_var': [],
            'archive_obj': [],
            'children_var': [],
            'children_obj': []
        }

        logging.info('seed: {}'.format(seed))
        start = time.time()

        archive_var, archive_obj = tree_search(args, log_run)
        end = time.time()

        archive_convert, archive_convert_norm = convert_archive(args, archive_var)
        indicators = get_indicators(args, archive_convert, archive_convert_norm)
        
        logging.info(indicators)

        log_run['total_time'] = end-start

        
        log_run['indicators'] = indicators
        log_run['archive_convert'] = archive_convert
        log_run['archive_convert_norm'] = archive_convert_norm
        

        pickle_out = open(os.path.join(args.save, f'seed_{seed}.pickle'), "wb")
        pickle.dump(log_run, pickle_out)
        pickle_out.close()
