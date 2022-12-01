# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/core.functions.ipynb.

# %% auto 0
__all__ = ['prepare_env']

# %% ../../nbs/core.functions.ipynb 3
import os
import torch.backends.cudnn as cudnn

from ..utils import fix_random_seed, backup_codes, rm

# %% ../../nbs/core.functions.ipynb 4
def prepare_env(cfg):
    # fix random seed
    fix_random_seed(cfg.BASIC.SEED)
    # cudnn
    cudnn.benchmark = cfg.CUDNN.BENCHMARK  # Benchmark will impove the speed
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC  #
    cudnn.enabled = cfg.CUDNN.ENABLE  # Enables benchmark mode in cudnn, to enable the inbuilt cudnn auto-tuner