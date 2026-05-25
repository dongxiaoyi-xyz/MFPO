# Mean Flow Policy Optimization (MFPO)

## Overview

This repository provides the implementation of the Mean Flow Policy Optimization (MFPO) algorithm.

## Installation

To get started, you need to install the required dependencies.

```bash
conda create -n MFPO python=3.9
conda activate MFPO
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Getting Started

To reproduce the results in the paper, execute the training script:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_online.py --config configs/mfpo_config.py --env_name Humanoid-v3
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_online.py --config configs/mfpo_config.py --env_name HalfCheetah-v3 --config.use_cdq False
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_online.py --config configs/mfpo_config.py --env_name Ant-v3
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_online.py --config configs/mfpo_config.py --env_name Walker2d-v3
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_online.py --config configs/mfpo_config.py --env_name Hopper-v3
```
When running with multiple gpus, the batch size (default 256) should be divisible by the number of devices.


The code is built based on the [QSM](https://github.com/Alescontrela/score_matching_rl) and [MaxEntDP](https://github.com/diffusionyes/MaxEntDP) implementation.
