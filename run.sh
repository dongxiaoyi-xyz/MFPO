XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_online.py --config configs/mfpo_config.py --env_name Humanoid-v3
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_online.py --config configs/mfpo_config.py --env_name HalfCheetah-v3 --config.use_cdq False
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_online.py --config configs/mfpo_config.py --env_name Ant-v3
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_online.py --config configs/mfpo_config.py --env_name Walker2d-v3
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_online.py --config configs/mfpo_config.py --env_name Hopper-v3