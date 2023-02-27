from copy import deepcopy
import random
from utils import *
from indicators import *
from nats_bench import create
from foresight.models.nasbench2 import get_model_from_arch_str
import logging
import numpy as np
from tqdm import tqdm
import time


from foresight.pruners import predictive
from foresight.weight_initializers import init_net
from foresight.models.nasbench2 import get_model_from_arch_str
from foresight.dataset import get_cifar_dataloaders

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar

import torch
import torch.optim as optim
import torch.nn.functional as F

from xautodl.utils.flop_benchmark import get_model_infos



def find_the_better(x, y):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    if isinstance(x[-1], dict):
        vote_lst = []
        for key in x[-1]:
            x_new = np.array([x[0], x[-1][key]])
            y_new = np.array([y[0], y[-1][key]])
            sub_ = x_new - y_new
            x_better = np.all(sub_ <= 0)
            y_better = np.all(sub_ >= 0)
            if x_better == y_better:  # True - True
                vote_lst.append(-1)
            elif y_better:  # False - True
                vote_lst.append(1)
            else:
                vote_lst.append(0)  # True - False
        count_vote_lst = [vote_lst.count(-1), vote_lst.count(0), vote_lst.count(1)]
        better_lst = np.array([-1, 0, 1])
        # if count_vote_lst[0] == count_vote_lst[1] == count_vote_lst[2] == 1:
        if count_vote_lst[0] == 1 or count_vote_lst[1] == 1 or count_vote_lst[2] == 1:
            return None
        idx = np.argmax(count_vote_lst)
        return better_lst[idx]
    else:
        sub_ = x - y
        x_better = np.all(sub_ <= 0)
        y_better = np.all(sub_ >= 0)
        if x_better == y_better:  # True - True
            return -1
        if y_better:  # False - True
            return 1
        return 0  # True - False

def remove_dominated(F_tm):
    F = F_tm.copy()
    F[:, 1] = 100 - F[:, 1]
    l = len(F)
    r = np.zeros(l, dtype=np.int8)
    for i in range(l):
        if r[i] == 0:
            for j in range(i + 1, l):
                better_sol = find_the_better(F[i], F[j])
                if better_sol == 0:
                    r[j] += 1
                elif better_sol == 1:
                    r[i] += 1
                    break
    return F_tm[r == 0]



def get_r0r1_front(F_obj_tm, F_var):
    F_obj = F_obj_tm.copy()
    l = len(F_obj)
    r = np.zeros(l, dtype=np.int8)
    for i in range(l):
        if r[i] == 0:
            for j in range(i + 1, l):
                better_sol = find_the_better(F_obj[i], F_obj[j])
                if better_sol == 0:
                    r[j] += 1
                elif better_sol == 1:
                    r[i] += 1
                    break
    F_obj_r0 = F_obj_tm[r == 0].copy()
    F_var_r0 = F_var[r == 0].copy()
    space_obj = F_obj_tm[np.invert(r == 0)].copy()
    space_var = F_var[np.invert(r == 0)].copy()
        
    return F_obj_r0, F_var_r0, space_obj, space_var


def is_dominated(ind_a, ind_b):
    n_obj = len(ind_a)
    greater = False

    for i in range(n_obj):
        if ind_b[i] > ind_a[i]: 
            return False
        if ind_b[i] < ind_a[i]:
            greater = True 

    return greater



def convert_archive(args, archive_var):
    archive_convert, archive_convert_norm = {}, {}
    for dataset in args.datasets:
        archive_convert[dataset], archive_convert_norm[dataset] = {}, {}
        archive_convert[dataset] = []        
        archive_convert_norm[dataset] = []
        for arch in archive_var:
            arch_str = subnet_to_str(arch)
            accuracy = evaluate_arch(arch_str=arch_str, dataset=dataset, args=args, measure='test-accuracy', epoch=200, in_bench=True)
            complexity = evaluate_arch(arch_str=arch_str, dataset=dataset, args=args, measure='flops', in_bench=True)
            archive_convert[dataset].append((complexity, accuracy))    
            

        archive_convert[dataset] = remove_dominated(np.array(archive_convert[dataset]))
        archive_convert_norm[dataset] = archive_convert[dataset].copy()
        archive_convert_norm[dataset][:, 0] = (archive_convert_norm[dataset][:, 0] - args.max_min_measures[dataset]['flops_min']) / (args.max_min_measures[dataset]['flops_max'] - args.max_min_measures[dataset]['flops_min'])
        archive_convert_norm[dataset][:, 1] = archive_convert_norm[dataset][:, 1] / 100
    return archive_convert, archive_convert_norm

def setup_experiment(net, args):

    optimiser = optim.SGD(
        net.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=0, last_epoch=-1)

    return optimiser, lr_scheduler

def train_nb2(arch_str, args):
    res = {'logmeasures': []}
    net = get_model_from_arch_str(arch_str, get_num_classes(args.dataset), init_channels=args.init_channels)
    net.to(args.device)

    optimiser, lr_scheduler = setup_experiment(net, args)

    train_loader = args.train_loader
    val_loader = args.val_loader
    
    #start training
    criterion = F.cross_entropy
    trainer = create_supervised_trainer(net, optimiser, criterion, args.device)
    evaluator = create_supervised_evaluator(net, {
        'accuracy': Accuracy(),
        'loss': Loss(criterion)
    }, args.device)

    # pbar = ProgressBar()
    # pbar.attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch(engine):
            
        #change LR
        lr_scheduler.step()

        #run evaluator
        evaluator.run(val_loader)

        #metrics
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']

        # pbar.log_message(f"Validation Results - Epoch: {engine.state.epoch} Avg accuracy: {round(avg_accuracy*100,2)}% Val loss: {round(avg_loss,2)} Train loss: {round(engine.state.output,2)}")

        measures = {}
       
        measures['train_loss'] = engine.state.output
        measures['val_loss'] = avg_loss
        measures['val_acc'] = avg_accuracy
        measures['epoch'] = engine.state.epoch
        res['logmeasures'].append(measures)

    #at epoch zero
    #run evaluator
    evaluator.run(val_loader)

    #metrics
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss']
    measures = {}
  
    measures['train_loss'] = 0
    measures['val_loss'] = avg_loss
    measures['val_acc'] = avg_accuracy
    measures['epoch'] = 0
    res['logmeasures'].append(measures)

    #run training
    stime = time.time()
    trainer.run(train_loader, args.epochs)
    etime = time.time()

    res['time'] = etime-stime
    return res

def evaluate_arch(arch_str, dataset, measure, args, in_bench=True, epoch=None, use_log=False):
    if measure in ['synflow', 'jacob_cov', 'snip']:
        net = get_model_from_arch_str(arch_str, get_num_classes(args))
        net.to(args.device)
        init_net(net, args.init_w_type, args.init_b_type)

        measures = predictive.find_measures(net, 
                                            args.train_loader, 
                                            (args.dataload, args.dataload_info, get_num_classes(dataset)), 
                                            args.device,
                                            measure_names=[measure])    
                                
        res = measures[measure]
        if np.isnan(res):
            res = -1e9


    elif measure=='valid-accuracy' and epoch==2:
        res = train_nb2(arch_str, args)['logmeasures'][epoch]['val_acc']
    

    elif in_bench:
        if dataset=='cifar10' and measure in ['valid-accuracy', 'train-all-time']:
            dataset='cifar10-valid'
            
        if measure.startswith(("train", "val", "test")):
            xinfo = args.api.get_more_info(
                arch_str,
                hp=epoch,
                dataset=dataset,
                is_random=False,
            )
            res = xinfo[measure]

                
        elif measure in ['flops']:
            arch_index = args.api.query_index_by_arch(arch_str)
            info = args.api.get_cost_info(arch_index, dataset)
            res = info[measure]
        elif measure=='train-all-time':
            arch_index = args.api.query_index_by_arch(arch_str)
            info = args.api.get_more_info(1234, 'cifar10', hp=epoch, is_random=False)
            res=info[measure]
    else: 
        if measure in ['flops']:
            net = get_model_from_arch_str(arch_str, get_num_classes(args))
            net.to(args.device)
            input_size = 16 if dataset == 'ImageNet16-120' else 32
            res, _ = get_model_infos(net, (len(args.train_loader), 3, input_size, input_size))

    return res

def get_child(args, node, pos, children):
    if pos == len(node):
        children.append(deepcopy(node)) 
    elif len(NODE_CHILDREN[args.method][node[pos]]) == 0:
        get_child(node, pos + 1, children) 
    else:
        for edge_type in NODE_CHILDREN[args.method][node[pos]]:
            cpy_node = deepcopy(node)
            cpy_node[pos] = edge_type
            get_child(args, cpy_node, pos + 1, children)

def tree_search(args, log_run):
    node = ['full', 'full', 'full', 'full', 'full', 'full']
    archive_var, archive_obj = [node], [None]
    
    for depth in range(0, 3):
        children_var = []
        
        for node in archive_var:
            get_child(args, node, 0, children_var)

        log_run['children_var'].append(children_var)
        

        children_obj = []
        for child_var in tqdm(children_var):
            start_arch = time.time()

            child_str = subnet_to_str(args, child_var)
            logging.info(child_var)
            in_bench=True if depth==2 else False
            child_obj = []
            child_obj.append(evaluate_arch(child_str, dataset=args.dataset, measure='flops', args=args, in_bench=in_bench))
            for proxy in args.objective_list[1:]:
                child_obj.append(-1 * evaluate_arch(child_str, dataset=args.dataset, measure=proxy, args=args, epoch=args.epochs, in_bench=in_bench))
            children_obj.append(child_obj)
            end_arch = time.time()
            log_run['time'].append(end_arch - start_arch)
        
        log_run['children_obj'].append(children_obj)
        logging.info(children_obj)
        children_obj = np.array(children_obj)
        children_var = np.array(children_var)
        if depth==2:
            archive_obj, archive_var,_, __ = get_r0r1_front(children_obj, children_var)
        else:
            archive_obj, archive_var,_, __ = get_r0r1_front(children_obj, children_var)

        log_run['archive_var'].append(archive_var)
        log_run['archive_obj'].append(archive_obj)
                
    
    return archive_var, archive_obj
