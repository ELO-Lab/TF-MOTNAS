import numpy as np
from utils import *
from search import evaluate_arch

from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV

def calc_IGD(archive, archive_norm, pf, pf_norm):
    get_igd = IGD(pf)
    get_igd_norm = IGD(pf_norm)

    igd = get_igd(archive)
    igd_norm = get_igd_norm(archive_norm)

    return igd, igd_norm

def calc_HV(archive_norm, ref_point):
    get_hv = HV(ref_point=ref_point)
    archive_norm_err = archive_norm.copy()
    archive_norm_err[:, 1] = 1 - archive_norm_err[:, 1]
    
    hv = get_hv(archive_norm_err)
    return hv

def calc_spread(archive_norm):
    idx_flops = archive_norm.argmin(axis=0)[0]
    idx_testacc = archive_norm.argmax(axis=0)[1]
    spread = np.linalg.norm(archive_norm[idx_flops] - archive_norm[idx_testacc])

    return spread
    

def get_indicators(args, archive_convert, archive_convert_norm):
    igd_dict, igd_norm_dict, hv_dict, spread_dict = {}, {}, {}, {}

    for dataset in args.datasets:
        igd_dict[dataset], igd_norm_dict[dataset], hv_dict[dataset],spread_dict[dataset] = {}, {}, {}, {}
        
        archive_temp = archive_convert[dataset]
        archive_temp_norm = archive_convert_norm[dataset]

        pf_temp = args.pf[dataset]['testacc-flops']
        pf_temp_norm = args.pf_norm[dataset]['testacc-flops']

        igd, igd_norm = calc_IGD(archive_temp, archive_temp_norm, pf_temp, pf_temp_norm)

        igd_dict[dataset] = igd
        igd_norm_dict[dataset] = igd_norm

        hv = calc_HV(archive_temp_norm, args.ref_point)
        hv_dict[dataset] = hv

        # spread = calc_spread(archive_temp_norm)
        # spread_dict[dataset] = spread

    return {'igd': igd_dict, "igd_norm": igd_norm_dict, "hv": hv_dict}

def evaluate_testacc(args, archive_var, archive_obj):
    index_zerocost_max = np.argmin(archive_obj, axis=0)
    testacc = {}
    total_training_time = []
    for dataset in args.datasets:
        testacc[dataset]  = []
        for arch_index in index_zerocost_max:
            arch_str = subnet_to_str(archive_var[arch_index])
            testacc[dataset].append(evaluate_arch(arch_str, dataset, 'test-accuracy', args, in_bench=True, epoch=200))
            if dataset==args.dataset:
                total_training_time.append(evaluate_arch(arch_str, args.dataset, 'train-all-time', args, in_bench=True, epoch=200))
    return testacc, total_training_time



  
