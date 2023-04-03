# Efficient Multi-Objective Neural Architecture Search via Tree Search with Training-Free Metrics
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)

An Vo, Nhat Minh Le, and Ngoc Hoang Luong

## Setup
- Clone this repository
- Install packages
```
$ pip install -r requirements.txt
```
- Download [NATS-Bench](https://drive.google.com/drive/folders/17S2Xg_rVkUul4KuJdq0WaWoUuDbo8ZKB), put it in the `benchmark` folder and follow instructions [here](https://github.com/D-X-Y/NATS-Bench)
## Usage
To run the code, use the command below with the required arguments

```shell
python search.py --method <method_name> --dataset <dataset_name> --n_runs <number_of_runs>
```

Refer to `main.py` for more details.
Example commands:
```shell
# TF-MOTNAS-A
python main.py --method TF-MOTNAS-A --dataset cifar10 --n_runs 30

# TF-MOTNAS-B
python main.py --method TF-MOTNAS-B --dataset cifar10 --n_runs 30
```


## Acknowledgement
Our source code is inspired by:
- [pymoo: Multi-objective Optimization in Python](https://github.com/anyoptimization/pymoo)
- [Zero-Cost Proxies for Lightweight NAS](https://github.com/SamsungLabs/zero-cost-nas)
- [NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size](https://github.com/D-X-Y/NATS-Bench)
- [Enhancing Multi-Objective Evolutionary Neural Architecture Search with Training-Free Pareto Local Search](https://github.com/ELO-Lab/MOENAS-TF-PSI)
