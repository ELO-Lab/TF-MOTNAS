import numpy as np
from utils import *
from mo_search import evaluate_arch

def archive_check(archive_var, archive_obj, ind_var, ind_obj):
    len_archive = len(archive_obj)
    dominated_indexes = []
    for i in range(len_archive):
        if is_dominated(ind_obj, archive_obj[i]):
            return False
        if is_dominated(archive_obj[i], ind_obj):
            dominated_indexes.append(i)

    for i in sorted(dominated_indexes, reverse=True):
        del archive_var[i]
        del archive_obj[i]
    
    archive_var.append(ind_var)
    archive_obj.append(ind_obj)

    return True

def remove_dominated(archive_convert):
    archive_convert_temp = archive_convert.copy()
    archive_convert_temp[:, 1] = 100 - archive_convert_temp[:, 1]
    is_efficient = np.ones(archive_convert_temp.shape[0], dtype=bool)
    for i, c in enumerate(archive_convert_temp):
        is_efficient[i] = np.all(np.any(archive_convert_temp[:i]>=c, axis=1)) and np.all(np.any(archive_convert_temp[i+1:]>=c, axis=1))
    return archive_convert[is_efficient]

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